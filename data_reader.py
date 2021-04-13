'''
Data Reader

April 2021
modric10zhang@gmail.com

'''
import sys
from id_allocator import IdAllocator

class DataReader(object):
    def __init__(self, batch_num):
        self._id_tool = IdAllocator()
        self._data = []
        self._batch = batch_num

    def parse_feature(self, field_info):
        pass

    def load(self, sample_path):
        with open(sample_path, 'r') as fp:
            for sinfo in fp:
                info = sinfo.strip().split('\t')
                for ii in info:
                    pos = ii.find(':')
                    k,v = ii[:pos], ii[pos+1:]
                    if 'uf' in k:
                        pass
                print("hello,world!")
                sys.exit(0)

dr = DataReader(125)

dr.load('sample.data')


'''
def gen_ff(fv, fea_limit):
    ff = set()
    if len(fv.strip()):
        ff = set([int(x) % fea_limit for x in fv.split(',') if int(x) % fea_limit > 0])
    if len(ff) == 0:
        ff = set([0])
    return ff
'''
'''
def get_feats(fs, fea_limit):
    feats = []
    for kv in fs.split(' '):
        pos = kv.find(':')
        if pos < 0:
            break
        kk, vv = kv[:pos], kv[pos+1:]
        feats.append(gen_ff(vv, fea_limit))
    return feats
'''
'''
def parse_line(line, user_field_num, sd_field_num, rd_field_num, con_field_num, fea_limit):    
    info = line.strip('\n').split('\t')
    if len(info) != 11:
        raise Exception('Sample Data Formate Error, len(info):%d, line=%s' % (len(info), line))
    names = ['uin', 'channel', 'sid', 'seed_doc', 'rele_doc']
    tag = True
    clk = 0.
    rwd = 0.
    uff, sdf, rdf, cff = [], [], [], []
    for kv in info:
        pos = kv.find(':')
        if pos < 0:
            tag = False
            break
        kk, vv = kv[:pos], kv[pos+1:]
        if kk in names:
            continue
        if 'uf' in kk:
            uff = get_feats(kv, fea_limit)
        elif 'sf' in kk:
            sdf = get_feats(kv, fea_limit)
        elif 'rf' in kk:
            rdf = get_feats(kv, fea_limit)
        elif 'cf' in kk:
            cff = get_feats(kv, fea_limit)
        elif kk == 'click':
            clk = float(vv)
        elif kk == 'reward':
            ori_rwd = float(vv)
            rwd = 0. if ori_rwd <= 3 else  1.0
        else:
            tag = False
            break
    if not tag or len(uff) != user_field_num or len(sdf) != sd_field_num or len(rdf) != rd_field_num or len(cff) != con_field_num:
        sys.stderr.write('parse rec list error,' + str(tag) + '--' + str(len(uff)) + ',' + str(len(sdf)) + ',' + str(len(rdf))+ ',' + str(len(cff)) + '\n')
        raise Exception('Parse Rec List Error!!!')
    return uff, sdf, rdf, cff, clk, rwd
'''
'''
def read_data(fname, user_field_num, sd_field_num, rd_field_num, con_field_num, fea_limit):
    rcnt = 0
    buf = []
    with open(fname, 'r') as fp:
        for line in fp:
            rcnt += 1
            if rcnt % 100 == 0:
                print datetime.datetime.now(), 'read ' + fname + ', lines =', rcnt
                sys.stdout.flush()
            try:
                uff, sdf, rdf, cff, clk, rwd = parse_line(line, user_field_num, sd_field_num, rd_field_num, con_field_num, fea_limit)
                #sid, offset, clk_seq, rec_list = parse_line(line, user_field_num, sd_field_num, rd_field_num, con_field_num, fea_limit)
            except Exception, e:
                sys.stderr.write('exception:' + str(e) + '\n')
                continue
            buf.append([uff, sdf, rdf, cff, clk, rwd])
            if len(buf) > 500:
                yield buf[:500]
                buf = buf[500:]
        if len(buf):
            yield buf
            buf = []
'''

#for pulls in read_data('sample.in', 5, 5, 5, 5, 8388593):
#    #print 'pull_num:', len(pulls)
#    for data in pulls:
#        print data[0]
#        print data[1]
#        print data[2]
#        print data[3]
#        print data[4]
#        print data[5]
#        break
#    break
