"""
compute variance given a set of embeddings
"""
import numpy as np
import pandas as pd
import tensorflow as tf

class metric_variance:

    def compute_variance(self, matrix):
        """
        use tf.reduce_mean to get 1d value of variance
        can also
        :param matrix:
        :return:
        """
        mean, var = tf.nn.moments(matrix, axes=0)
        res = tf.reduce_mean(var)
        return res

    def compute_standard_variance(self, matrix):
        """
        compute standard variance to avoid changing of dimension scale
        :param matrix:
        :return:standard variance of embs
        """
        mean, var = tf.nn.moments(matrix, axes=0)
        nomorlized_var = tf.sqrt(var)
        res = tf.reduce_mean(nomorlized_var)
        return res

    def get_compute_matrix(self, emb_list):
        res = emb_list[0]
        for i in range(1, len(emb_list)):
            res = tf.concat([res, emb_list[i]], axis=0)
        return res

# embs = []
# a = np.random.rand(12,24)
# b = np.random.rand(12,24)
# c = np.random.rand(12,24)
# embs.append(a)
# embs.append(b)
# embs.append(c)
# metric = metric_variance()
# res = metric.get_compute_matrix(embs)
# # print(res.shape)