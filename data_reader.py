'''
Data Reader

April 2021
modric10zhang@gmail.com

'''
import sys
from id_allocator import IdAllocator
from hyper_param import param_dict as pd

class DataReader(object):
    def __init__(self, batch_num):
        self._id_tool = IdAllocator()
        self._data = []
        self._batch = batch_num

    def unique_feature_num(self):
        return self._id_tool.unique_id_num()

    def parse_feature(self, raw_feature):
        feature = set()
        for f in raw_feature.split(','):
            feature.add(self._id_tool.allocate(f))
        if len(feature) == 0:
            feature.add(0)
        return feature

    def load(self, sample_path):
        with open(sample_path, 'r') as fp:
            for sinfo in fp:
                skv = {}
                info = sinfo.strip().split('\t')
                for ii in info:
                    ff = ii.split(' ')
                    for fi in ff:
                        pos = fi.find(':')
                        skv[fi[:pos]] = fi[pos+1:]
                feats = [[], [], [], []]
                fields = [pd['user_field_num'], pd['doc_field_num'], pd['con_field_num'], pd['doc_field_num']]
                prefix = ['uf', 'rf', 'cf', 'sf']
                assert(len(fields) == len(prefix))
                for k in range(len(fields)):
                    for i in range(fields[k]):
                        fk = '%s%s' % (prefix[k], i)
                        if fk not in skv:
                            raise Exception('field %s not exist.' % fk)
                        feats[k].append(self.parse_feature(skv[fk]))
                self._data.append([feats[0], feats[1], feats[2], feats[3], skv['dwell_time']])

    def next(self):
        nb = None
        if len(self._data) <= 0:
            return nb
        else:
            idx = len(self._data) if len(self._data) <= self._batch else self._batch
            nb = self._data[:idx]
            self._data = self._data[idx:]
        return nb

