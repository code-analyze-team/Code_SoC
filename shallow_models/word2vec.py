import collections
import math
import os
from six.moves import urllib
import zipfile
import random
import re
import json

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from six.moves import xrange  # pylint: disable=redefined-builtin


# Set random seeds
SEED = 2019
random.seed(SEED)
np.random.seed(SEED)


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

class Word2Vec(BaseEstimator, TransformerMixin):
    """
    things to select:
    1. which data file to use
    2. model: cbow, skipgram, cbow with CNN
    3. loss
    4. optimizer
    """
    def __init__(self, batch_size=128, num_skips=2, skip_window=1,
                 architecture='skip-gram', embedding_size=128, vocabulary_size=50000,
                 loss_type='sampled_softmax_loss', n_neg_samples=64,
                 optimizer='SGD', learning_rate=1.0, epochs=1000001,
                 valid_size=16, valid_window=100, data_mode='text8'):

        # bind training params to class
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.architecture = architecture
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.loss_type = loss_type
        self.n_neg_samples = n_neg_samples
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.valid_size = valid_size
        self.valid_window = valid_window

        # set up data params
        self.data_mode = data_mode
        self.glove_path = '/home/qwe/zfy_data/SoC/data_new/nl_pre/glove/glove.6B.100d.txt'
        self._set_data_paths()

        # set up global variables
        self.data_index = 0
        self._build_dictionaries()
        self._choose_batch_generator()
        self._pick_valid_samples()
        self._init_graph()

    def _set_data_paths(self):
        if self.data_mode == 'random_walk':
            self.data_path = '/home/qwe/zfy_data/SoC/data_new/random_walks#with_edge#union#reduced'
            self.log_dir = '/home/qwe/zfy_data/SoC/data_new/random_walk'
        elif self.data_mode == 'nl_tokens':
            self.data_path = '/home/qwe/zfy_data/SoC/data_new/nl#with_edge#union#reduced'
            self.log_dir = '/home/qwe/zfy_data/SoC/data_new/nl_pre'
        elif self.data_mode == 'text8':
            self.data_path = '/home/qwe/zfy_data/SoC/data_new/text8/text8.zip'
            self.log_dir = '/home/qwe/zfy_data/SoC/data_new/text8'
            if not os.path.exists(self.data_path):
                self.data_path, _ = urllib.request.urlretrieve('http://mattmahoney.net/dc/' + 'text8.zip', self.data_path)
        else:
            self.data_path = None


    def _pick_valid_samples(self):
        valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))
        self.valid_examples = valid_examples

    def _build_dictionaries(self):
        if self.data_mode == 'random_walk':
            with open(self.data_path) as f:
                words = f.read().split('#')
            print('len: ', len(words))
        elif self.data_mode == 'nl_tokens':
            with open(self.data_path) as f:
                words = f.read().split(' ')
            print('len: ', len(words))
        elif self.data_mode == 'text8':
            with zipfile.ZipFile(self.data_path) as f:
                words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        else:
            words = None

        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.vocabulary_size))
        dictionary = {}
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = []
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        self.data = data
        self.count = count
        self.dictionary = dictionary
        self.reversed_dictionary = reversed_dictionary
        # self.vocabulary_size = len(count)

    def _choose_batch_generator(self):
        if self.architecture == 'skip-gram':
            self.generate_batch = self._generate_batch_skipgram
        elif self.architecture == 'cbow':
            self.generate_batch = self._generate_batch_cbow

    def _generate_batch_skipgram(self):
        # print('index: ', self.data_index)
        assert self.batch_size % self.num_skips == 0
        assert self.num_skips <= 2 * self.skip_window
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
        if self.data_index + span > len(self.data):
            self.data_index = 0
        buffer.extend(self.data[self.data_index:self.data_index + span])
        self.data_index += span
        for i in range(self.batch_size // self.num_skips):
            context_words = [w for w in range(span) if w != self.skip_window]
            words_to_use = random.sample(context_words, self.num_skips)
            #         print(context_words, words_to_use)
            for j, context_word in enumerate(words_to_use):
                batch[i * self.num_skips + j] = buffer[self.num_skips]
                labels[i * self.num_skips + j, 0] = buffer[context_word]
            if self.data_index == len(self.data):
                buffer.extend(self.data[0:span])
                self.data_index = span
            else:
                buffer.append(self.data[self.data_index])
                self.data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        self.data_index = (self.data_index + len(self.data) - span) % len(self.data)
        return batch, labels

    def _generate_batch_cbow(self):
        return None

    # def _init_graph(self):
    #     self.graph = tf.Graph()
    #     with self.graph.as_default():
    #         # Set graph level random seed
    #         tf.set_random_seed(SEED)
    #         # Input data.
    #         if self.architecture == 'skip-gram':
    #             self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
    #         elif self.architecture == 'cbow':
    #             self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_skips])
    #         self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
    #         self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)
    #
    #         # Variables.
    #         self.embeddings = tf.Variable(
    #             tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
    #
    #         self.weights = tf.Variable(
    #             tf.truncated_normal([self.vocabulary_size, self.embedding_size],
    #                                 stddev=1.0 / math.sqrt(self.embedding_size)))
    #
    #         self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))
    #
    #         # Model.
    #         # Look up embeddings for inputs.
    #         if self.architecture == 'skip-gram':
    #             self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_dataset)
    #         elif self.architecture == 'cbow':
    #             embed = tf.zeros([self.batch_size, self.embedding_size])
    #             for j in range(self.num_skips):
    #                 embed += tf.nn.embedding_lookup(self.embeddings, self.train_dataset[:, j])
    #             self.embed = embed
    #
    #         # Compute the loss, using a sample of the negative labels each time.
    #         if self.loss_type == 'sampled_softmax_loss':
    #             loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels, self.embed, self.n_neg_samples, self.vocabulary_size)
    #         elif self.loss_type == 'nce_loss':
    #             loss = tf.nn.nce_loss(self.weights, self.biases, self.train_labels, self.embed, self.n_neg_samples, self.vocabulary_size)
    #         else:
    #             loss = None
    #         self.loss = tf.reduce_mean(loss)
    #
    #         # Optimizer.
    #         if self.optimizer == 'Adagrad':
    #             self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
    #         elif self.optimizer == 'SGD':
    #             self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
    #
    #         # Compute the similarity between minibatch examples and all embeddings.
    #         # We use the cosine distance:
    #         norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
    #         self.normalized_embeddings = self.embeddings / norm
    #         self.valid_embeddings = tf.nn.embedding_lookup(
    #             self.normalized_embeddings, self.valid_dataset)
    #         self.similarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddings))
    #
    #         # init op
    #         self.init_op = tf.initialize_all_variables()
    #         # create a saver
    #         self.saver = tf.train.Saver()

    def _init_graph(self):
        graph = tf.Graph()
        # Input data.
        with graph.as_default():
            with tf.name_scope('inputs'):
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
                valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                # Look up embeddings for inputs.
                with tf.name_scope('embeddings'):
                    embeddings = tf.Variable(
                        tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(
                    tf.truncated_normal(
                        [self.vocabulary_size, self.embedding_size],
                        stddev=1.0 / math.sqrt(self.embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.nce_loss(
                        weights=nce_weights,
                        biases=nce_biases,
                        labels=train_labels,
                        inputs=embed,
                        num_sampled=self.n_neg_samples,
                        num_classes=self.vocabulary_size))

            # Add the loss value as a scalar to summary.
            tf.summary.scalar('loss', loss)

            # Construct the SGD optimizer using a learning rate of 1.0.
            with tf.name_scope('optimizer'):
                optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            # Compute the cosine similarity between minibatch examples and all embeddings.
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                                      valid_dataset)
            similarity = tf.matmul(
                valid_embeddings, normalized_embeddings, transpose_b=True)

            # Merge all summaries.
            merged = tf.summary.merge_all()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            # Create a saver.
            saver = tf.train.Saver()

        num_steps = 100001

        with tf.Session(graph=graph) as session:
            # Open a writer to write summaries.
            writer = tf.summary.FileWriter(self.log_dir, session.graph)

            # We must initialize all variables before we use them.
            init.run()
            print('Initialized')

            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(self.data, self.batch_size, self.num_skips, self.skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # Define metadata variable.
                run_metadata = tf.RunMetadata()

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
                # Feed metadata variable to session for visualizing the graph in TensorBoard.
                _, summary, loss_val = session.run(
                    [optimizer, merged, loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                # print('loss: ', loss_val)
                average_loss += loss_val

                # Add returned summaries to writer in each step.
                writer.add_summary(summary, step)
                # Add metadata to visualize the graph for the last run.
                if step == (num_steps - 1):
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                # Note that this is expensive (~20% slowdown if computed every 500 steps)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(self.valid_size):
                        valid_word = self.reversed_dictionary[self.valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = self.reversed_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)
            final_embeddings = normalized_embeddings.eval()

            # Write corresponding labels for the embeddings.
            with open(self.log_dir + '/metadata.tsv', 'w') as f:
                for i in xrange(self.vocabulary_size):
                    f.write(self.reversed_dictionary[i] + '\n')

            # Save the model for checkpoints.
            saver.save(session, os.path.join(self.log_dir, 'model.ckpt'))

        writer.close()

    # def fit(self):
    #     # with self.sess as session:
    #     session = self.sess
    #     writer = tf.summary.FileWriter(self.log_dir, session.graph)
    #     session.run(self.init_op)
    #
    #     average_loss = 0
    #     print("Initialized")
    #     for step in range(self.epochs):
    #         batch_data, batch_labels = self.generate_batch()
    #         # print(self.data_index)
    #         feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
    #         op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
    #         print(l)
    #         # average_loss += l
    #         # if step % 2000 == 0:
    #         #     if step > 0:
    #         #         average_loss = average_loss / 2000
    #         #     # The average loss is an estimate of the loss over the last 2000 batches.
    #         #     print('Average loss at step %d: %f' % (step, average_loss))
    #         #     average_loss = 0
    #         # note that this is expensive(~20 % slowdown if computed every 500 steps)
    #         with session.as_default():
    #             if step % 10000 == 0:
    #                 sim = self.similarity.eval(session=self.sess)
    #                 for i in range(self.valid_size):
    #                     valid_word = self.reversed_dictionary[self.valid_examples[i]]
    #                     top_k = 8  # number of nearest neighbors
    #                     nearest = (-sim[i, :]).argsort()[1:top_k + 1]
    #                     log = 'Nearest to %s:' % valid_word
    #                     for k in range(top_k):
    #                         close_word = self.reversed_dictionary[nearest[k]]
    #                         log = '%s %s,' % (log, close_word)
    #                     print(log)
    #
    #     # final_embeddings = self.normalized_embeddings.eval()
    #     final_embeddings = session.run(self.normalized_embeddings)
    #     self.final_embeddings = final_embeddings

w2v = Word2Vec()
# print(len(w2v.count))
# print('Most common words (+UNK)', w2v.count[:5])
# print('Sample data', w2v.data[:10], [w2v.reversed_dictionary[i] for i in w2v.data[:10]])
# print(len(w2v.data))
