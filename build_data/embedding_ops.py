import tensorflow as tf
import numpy as np
import pandas as pd

"""
look up embedding given a embedding matrix
"""

class emb_ops:

    def __init__(self, tsv_path):
        """
        set up embedding matrix from embedding.tsv
        for embedding lookup
        """
        df = pd.read_csv(tsv_path, header=None, sep=' ')
        matrix = df.values
        unk_mat = np.random.rand(1,128)
        unk_mat = (unk_mat - 0.5) / 10 # so that value is in [-0.5, 0.5]
        m = np.concatenate((unk_mat, matrix), axis=0)
        self.matrix = m

    def lookup_single(self, index):
        """
        given a single index
        return lookup result in self.emb_tensor
        """
        input = [index]
        return tf.nn.embedding_lookup(self.matrix, input)

    def lookup_1d(self, indexes):
        """
        given a list of indexes
        return lookup result in self.emb_tensor
        """
        return tf.nn.embedding_lookup(self.matrix, indexes)

