'''
Model Hyper Parameter Dict

April 2021
modric10zhang@gmail.com

'''

param_dict = {
    'feat_dim'          :       6,              # feature embedding dimension
    'user_field_num'    :       5,              # number of user feature fields
    'doc_field_num'     :       5,              # number of doc feature fields
    'con_field_num'     :       5,              # number of context feature fields
    'expert_dim'        :       90,             # expert subnetwork dimension
    'expert_num'        :       3,              # number of experts
    'critic_num'        :       4,              # number of critic networks(gates)
    'num_epochs'        :       100,            # training epoch
    'batch_size'        :       128,            # batch size
    'lr'                :       0.0002,         # learning rate of network
    'dropout'           :       0.3,            # dropout ratio
    'grad_clip'         :       5.0,            # grad clip
    'head_num'          :       3,              # head number for all self-attention units
}
