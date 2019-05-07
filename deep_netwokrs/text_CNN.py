"""
implementation of CNN model to oprate on texts
"""

import tensorflow as tf
import math

class CNN:
    """
    a CNN model to get representations of tokens
    --input: sentences, aka. sequences of words
    --embedding layer
    --multi conv layer
    --pooling
    --full connection
    --manually drop-out
    --negative-sampling
    """

    def __init__(self, sentence_length, vocab_size, embedding_size, filter_sizes, num_filters, num_labels=1, l2_reg_lambda=0.0):
        """
        init the class
        :param sentence_length: sentence length
        :param vocab_size: how many different words
        :param embedding_size: 200, 100 for structure ; 100 for natural language
        :param filter_sizes: n-grams
        :param num_filters: number of conv filters
        :param l2_reg_lambda: 0.0 means no l2 regulation
        """

        # configure word2vec parameters
        self.num_sampled = 64


        # input, output, dropout
        self.batch = tf.placeholder(tf.int32, [None, sentence_length], name='batch')  # place holder for word ids
        self.labels = tf.placeholder(tf.float32, [None, num_labels], name='label')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')  # value should be in [0, 1]

        # l2 regulation loss (optional)
        l2_loss = tf.constant(l2_reg_lambda)

        # embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')

            # result tensor: [None, sentence_length, embedding_size]
            self.embedded_words = tf.nn.embedding_lookup(self.W, self.batch)

            # since conv2d operation expects 4-dimensional tensor which are batch, width, height, channel respectively
            # we expand the embedding with chanel=1
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)

            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sentence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

            # constructs variables for nce loss estimate
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal([vocab_size, embedding_size],
                                        stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocab_size]))

            #
            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=self.labels,
                        inputs=self.h_pool_flat,
                        num_sampled=self.num_sampled,
                        num_classes=vocab_size))

    def train(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(

            )

