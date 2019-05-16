import tensorflow as tf
import numpy as np
import pandas as pd

"""
look up embedding given a embedding matrix
"""

class emb_ops:

    def __init__(self, glove_path):
        # """
        # set up embedding matrix from embedding.tsv
        # for embedding lookup
        # """
        # df = pd.read_csv(tsv_path, header=None, sep=' ')
        # matrix = df.values
        # unk_mat = np.random.rand(1,128)
        # unk_mat = (unk_mat - 0.5) / 10 # so that value is in [-0.5, 0.5]
        # m = np.concatenate((unk_mat, matrix), axis=0)
        # self.matrix = m

        map = {}
        f = open(glove_path)
        for line in f.readlines():
            split_line = line.split()
            word = split_line[0]
            emb = np.array([float(val) for val in split_line[1:]])
            emb = emb.reshape((1, -1))
            # print(emb.shape)
            map.setdefault(word, '')
            map[word] = emb
        self.glove = map
        self.label_mean = None


    def lookup_glove_embedding_single(self, target):
        emb = None
        if target in self.glove.keys():
            emb = self.glove.get(target)
        else:
            unk_mat = np.random.rand(1, 200)
            unk_mat = (unk_mat - 0.5) / 10  # so that value is in [-0.5, 0.5]
            emb = unk_mat
        assert emb.shape[0] > 0 and emb.shape[1] > 0
        return emb

    def lookup_glove_embedding_batch(self, tokens):
        emb = self.lookup_glove_embedding_single(tokens[0])
        for i in range(1, len(tokens)):
            tmp = self.lookup_glove_embedding_single(tokens[i])
            print('tmp: ', tmp.shape)
            emb = np.concatenate((emb, tmp), axis=0)
        assert emb.shape[0] == len(tokens)
        return emb

    def lookup(self, index):
        """
        lookup matrix built in init method.
        index can be single value or a multi - dimension matrix
        eg.: index = 5 or index = [2,4] or higher dimension
        """
        return tf.nn.embedding_lookup(self.matrix, index)


