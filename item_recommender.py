'''
Item Recommender

April 2021
modric10zhang@gmail.com

'''

import os
import math
import numpy as np
import tensorflow.compat.v1 as tf
from layer_util import *
from data_reader import DataReader
from hyper_param import param_dict as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.disable_eager_execution()

###### global variable for local computation ######
g_loss_sum = 0.
g_loss_cnt = 0

g_working_mode = 'local_train'
g_training = False

g_dr = DataReader(pd['batch_size'])


class ItemRecommender(object):
    def __init__(self):
        # placeholder
        self.sph_user = tf.sparse_placeholder(tf.int32, name='sph_user')
        self.sph_doc = tf.sparse_placeholder(tf.int32, name='sph_doc')
        self.sph_con = tf.sparse_placeholder(tf.int32, name='sph_con')
        self.sph_seed = tf.sparse_placeholder(tf.int32, name='sph_seed')
        self.sph_ig = tf.sparse_placeholder(tf.int32, name='sph_ig')
        self.ph_dwell_time = tf.placeholder(tf.float32, name='ph_dwell_time')

        self.create_graph('m3oe')
        diff = tf.reshape(self.ph_dwell_time, [-1]) - tf.reshape(self.output, [-1])
        self.loss = tf.reduce_mean(tf.square(diff))
        vs = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='m3oe')
        self.grads = tf.clip_by_global_norm(tf.gradients(self.loss, vs), pd['grad_clip'])[0]
        with tf.variable_scope('opt'):
            optimizer = tf.train.AdamOptimizer(pd['lr'])
            self.opt = optimizer.apply_gradients(zip(self.grads, vs))

    def field_interact(self, fields):
        global g_training
        qkv = tf.layers.dropout(fields, rate=pd['dropout'], training=g_training)
        with tf.variable_scope('fi'):
            return multihead_attention(queries=qkv,
                                       keys=qkv,
                                       values=qkv,
                                       num_heads=pd['head_num'],
                                       dropout_rate=pd['dropout'],
                                       training=g_training,
                                       causality=False,
                                       scope='mha')

    def create_graph(self, scope):
        global g_training, g_dr
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            feat_dict = get_embeddings(g_dr.unique_feature_num(),
                                       pd['feat_dim'],
                                       scope='feat_embedding',
                                       zero_pad=False)
            n_batch = pd['batch_size']
            n_user, n_doc, n_con = pd['user_field_num'], pd['doc_field_num'], pd['con_field_num']
            embed_dim = pd['feat_dim']
            user_embed = tf.nn.embedding_lookup_sparse(feat_dict, self.sph_user, sp_weights=None, combiner='mean')
            self.user = tf.reshape(user_embed, shape=[n_batch, n_user, embed_dim])
            doc_embed = tf.nn.embedding_lookup_sparse(feat_dict, self.sph_doc, sp_weights=None, combiner='mean')
            self.doc = tf.reshape(doc_embed, shape=[n_batch, n_doc, embed_dim])
            con_embed = tf.nn.embedding_lookup_sparse(feat_dict, self.sph_con, sp_weights=None, combiner='mean')
            self.con = tf.reshape(con_embed, shape=[n_batch, n_con, embed_dim])
            seed_embed = tf.nn.embedding_lookup_sparse(feat_dict, self.sph_seed, sp_weights=None, combiner='mean')
            self.seed = tf.reshape(seed_embed, shape=[n_batch, n_doc, embed_dim])
            ig_embed = tf.nn.embedding_lookup_sparse(feat_dict, self.sph_ig, sp_weights=None, combiner='mean')
            self.ig = tf.reshape(ig_embed, shape=[n_batch, n_doc, embed_dim])

            fi_in = tf.concat([self.doc, self.seed], axis=1)
            # feature interaction network
            fi_expert = tf.reshape(self.field_interact(fi_in), shape=[n_batch, -1])
            fi_expert = tf.concat([fi_expert,
                                   tf.reshape(self.user, shape=[n_batch, -1]),
                                   tf.reshape(self.con, shape=[n_batch, -1])], axis=1)
            fi_expert = tf.layers.dense(fi_expert, fi_expert.get_shape().as_list()[-1], activation=tf.nn.relu)
            fi_expert = tf.layers.dense(fi_expert, pd['expert_dim'], activation=tf.nn.relu)
            # sys.exit(0)
            edc = tf.reshape(self.doc, shape=[-1, embed_dim])
            esd = tf.reshape(self.seed, shape=[-1, embed_dim])
            # similarity network
            smn0 = tf.multiply(edc, esd)
            smn1 = tf.reduce_sum(tf.multiply(edc, esd), axis=1, keep_dims=True)
            smn = tf.reshape(tf.concat([smn0, smn1], axis=1), shape=[n_batch, -1])
            sim_expert = tf.concat([smn,
                                    tf.reshape(self.user, shape=[n_batch, -1]),
                                    tf.reshape(self.con, shape=[n_batch, -1])], axis=1)
            sim_expert = tf.layers.dense(sim_expert, pd['expert_dim'], activation=tf.nn.relu)
            # information gain network
            ig_expert = tf.concat([tf.reshape(self.ig, [n_batch, -1]),
                                   tf.reshape(self.user, [n_batch, -1]),
                                   tf.reshape(self.con, [n_batch, -1])], axis=1)
            ig_expert = tf.layers.dense(ig_expert, pd['expert_dim'], activation=tf.nn.relu)
            # multi-ciritic
            gate_in = tf.concat([tf.reshape(self.user, [n_batch, -1]),
                                 tf.reshape(self.seed, [n_batch, -1]),
                                 tf.reshape(self.con, [n_batch, -1])], axis=1)
            experts = tf.stack([fi_expert, sim_expert, ig_expert], axis=1)
            gates, votes = [], []
            for i in range(pd['critic_num']):
                gates.append(tf.nn.softmax(tf.layers.dense(gate_in, pd['expert_num'])))
                gates[i] = tf.reshape(gates[i], [n_batch, pd['expert_num'], 1])
                votes.append(tf.reduce_sum(gates[i] * experts, axis=1))
            votes = tf.stack(votes, axis=1)
            # attention layer
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            att_x = tf.concat([tf.reshape(self.user, [n_batch, -1]),
                               tf.reshape(self.doc, [n_batch, -1]),
                               tf.reshape(self.seed, [n_batch, -1]),
                               tf.reshape(self.con, [n_batch, -1])], axis=1)
            att_w = tf.get_variable('att_w', (pd['expert_dim'], att_x.get_shape().as_list()[-1]), initializer=w_init)
            att_o = tf.tensordot(votes, att_w, [[2], [0]])
            att_x = tf.tile(tf.expand_dims(att_x, 1), [1, pd['critic_num'], 1])
            att_o = tf.expand_dims(tf.nn.softmax(tf.reduce_sum(att_o * att_x, 2)), -1)
            vote_ret = tf.reduce_sum(att_o * votes, axis=1)
            fc = tf.layers.dropout(
                tf.layers.dense(vote_ret, vote_ret.get_shape().as_list()[-1] / 2, activation=tf.nn.relu),
                rate=pd['dropout'],
                training=g_training)
            self.output = tf.layers.dense(fc, 1, activation=tf.nn.relu)

    # call for evaluation
    def predict(self, sess, ph_dict):
        return sess.run(self.output, feed_dict={self.sph_user: ph_dict['user'],
                                                self.sph_doc: ph_dict['doc'],
                                                self.sph_con: ph_dict['con'],
                                                self.sph_seed: ph_dict['seed'],
                                                self.sph_ig: ph_dict['ig'],
                                                self.ph_dwell_time: ph_dict['reward']})

    # call for learning from data
    def learn(self, sess, ph_dict):
        loss, _ = sess.run([self.loss, self.opt], feed_dict={self.sph_user: ph_dict['user'],
                                                             self.sph_doc: ph_dict['doc'],
                                                             self.sph_con: ph_dict['con'],
                                                             self.sph_seed: ph_dict['seed'],
                                                             self.sph_ig: ph_dict['ig'],
                                                             self.ph_dwell_time: ph_dict['reward']})
        global g_loss_sum, g_loss_cnt
        g_loss_sum += np.mean(loss)
        g_loss_cnt += 1


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(max(min(-x, 1e2), -1e2)))


def handle(sess, net, sess_data):
    def gen_sparse_tensor(fs):
        global g_dr
        kk, vv = [], []
        for i in range(len(fs)):
            ff = fs[i]
            assert (isinstance(ff, set))
            ff = list(ff)
            for k in range(len(ff)):
                kk.append(np.array([i, k], dtype=np.int32))
                vv.append(ff[k])
        return tf.SparseTensorValue(kk, vv, [len(fs), g_dr.unique_feature_num()])

    if len(sess_data) != pd['batch_size']:
        return
    user, doc, con, seed, dwell = [], [], [], [], []
    for i in range(len(sess_data)):
        user.append(sess_data[i][0])
        doc.append(sess_data[i][1])
        con.append(sess_data[i][2])
        seed.append(sess_data[i][3])
        dwell.append(sess_data[i][4])
    phd = {}
    # print np.array(user).shape
    user = np.array(user).reshape(pd['batch_size'] * pd['user_field_num'])
    phd['user'] = gen_sparse_tensor(user)
    doc = np.array(doc).reshape(pd['batch_size'] * pd['doc_field_num'])
    phd['doc'] = gen_sparse_tensor(doc)
    seed = np.array(seed).reshape(pd['batch_size'] * pd['doc_field_num'])
    phd['seed'] = gen_sparse_tensor(seed)
    ig = []
    for i in range(doc.shape[0]):
        ig.append({0} if doc[i] <= seed[i] else doc[i] - seed[i])
    ig = np.array(ig).reshape(pd['batch_size'] * pd['doc_field_num'])
    phd['ig'] = gen_sparse_tensor(ig)
    con = np.array(con).reshape(pd['batch_size'] * pd['con_field_num'])
    phd['con'] = gen_sparse_tensor(con)
    phd['reward'] = dwell
    global g_training
    if g_training:
        # train network
        net.learn(sess, phd)
    else:
        # evaluate network
        qout = net.predict(sess, phd).reshape([-1])
        global g_working_mode
        for i in range(len(dwell)):
            if 'local_predict' == g_working_mode:
                print('%s %s' % (dwell[i], qout[i]))


def work():
    sess = tf.Session()
    # build networks
    net = ItemRecommender()
    saver = tf.train.Saver(max_to_keep=1)
    g_init_op = tf.global_variables_initializer()
    if os.path.exists('./ckpt') and len(os.listdir('./ckpt')):
        model_file = tf.train.latest_checkpoint('./ckpt')
        saver.restore(sess, model_file)
    else:
        sess.run(g_init_op)
        os.system('mkdir ckpt')
    global g_loss_sum, g_loss_cnt, g_dr
    last_epoch_loss = 1e2
    for k in range(pd['num_epochs']):
        if k > 0:
            g_dr.load('sample.data')
        data = g_dr.next()
        batch_cnt = 0
        while data is not None:
            handle(sess, net, data)
            data = g_dr.next()
            batch_cnt += 1
            if g_training and batch_cnt % 10 == 0:
                print('>>>Average Loss --- epoch %d --- batch %d --- %f' % (
                    k, batch_cnt, g_loss_sum / (g_loss_cnt + 1e-6)))
        print('>>>Average Loss --- epoch %d --- batch %d --- %f' % (k, batch_cnt, g_loss_sum / (g_loss_cnt + 1e-6)))
        if g_loss_sum / g_loss_cnt > last_epoch_loss:
            print('Job Finished!')
            break
        else:
            last_epoch_loss = g_loss_sum / g_loss_cnt
    saver.save(sess, 'ckpt/m3oe.ckpt')


if __name__ == '__main__':
    g_dr.load('sample.data')
    if g_working_mode == 'local_train':
        g_training = True
    elif g_working_mode == 'local_predict':
        g_training = False
    else:
        raise Exception('invalid working mode')
    work()
