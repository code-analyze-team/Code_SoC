import numpy as np
import collections
import random
from itertools import compress
from build_data.global_values import global_value

"""
a class to build data for word2vec model
"""
"""
deprecated for now since this part is running on jupyter notebook
"""

# class w2v_builder:
#
#     def __init__(self):
#         """
#         to init the class, feed with a pandas form text data set
#         eg. text8, tweet data set
#         :param data: a data set in pandas form
#         """
#         got = global_value()
#         data_index = 0
#         got.set_value('data_index', data_index)
#         self.got = got
#
#     def read_local_data(self, filename):
#         with open(filename) as f:
#             data = f.read().split()
#         return data
#
#     def preprocess(self, text):
#         text = text.lower()
#         text = text.replace(',', '')
#         text = text.replace('"', '')
#         text = text.replace('\"\"', '')
#         text = text.replace(';', '')
#         text = text.replace('!', '')
#         text = text.replace('?', '')
#         text = text.replace('(', '')
#         text = text.replace(')', '')
#         text = text.replace('--', '')
#         text = text.replace('?', '')
#         text = text.replace('\n', '')
#         text = text.replace(': ', ':')
#         text = text.replace('>>', '>')
#         text = text.replace(' ', ':')
#         return text
#
#     def build_dataset(self, words, vocabulary_size=50000):
#         '''
#         Build the dictionary and replace rare words with UNK token.
#
#         Parameters
#         ----------
#         words: list of tokens
#         vocabulary_size: maximum number of top occurring tokens to produce,
#         rare tokens will be replaced by 'UNK'
#         '''
#         count = [['UNK', -1]]
#         count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
#         dictionary = dict()  # {word: index}
#         data = list()  # collect index
#         unk_count = 0
#         for word, _ in count:
#             dictionary[word] = len(dictionary)
#         for word in words:
#             if word in dictionary:
#                 index = dictionary[word]
#             else:
#                 index = 0  # dictionary['UNK']
#                 unk_count += 1
#             data.append(index)
#         count[0][1] = unk_count  # list of tuples (word, count)
#         reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
#         return data, count, dictionary, reverse_dictionary
#
#     def generate_batch_skipgram(self, data, batch_size, num_skips, skip_window):
#         '''
#         Batch generator for skip-gram model.
#
#         Parameters
#         ----------
#         data: list of index of words
#         batch_size: number of words in each mini-batch
#         num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
#         skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
#         '''
#         # global data_index
#         data_index = self.got.get_value('data_index', -1)
#         assert batch_size % num_skips == 0
#         assert num_skips <= 2 * skip_window
#         batch = np.ndarray(shape=(batch_size), dtype=np.int32)
#         labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#         span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#         buffer = collections.deque(maxlen=span)  # used for collecting data[data_index] in the sliding window
#         for _ in range(span):
#             buffer.append(data[data_index])
#             data_index = (data_index + 1) % len(data)
#         for i in range(batch_size // num_skips):
#             target = skip_window  # target label at the center of the buffer
#             targets_to_avoid = [skip_window]
#             for j in range(num_skips):
#                 while target in targets_to_avoid:
#                     target = random.randint(0, span - 1)
#                 targets_to_avoid.append(target)
#                 batch[i * num_skips + j] = buffer[skip_window]
#                 labels[i * num_skips + j, 0] = buffer[target]
#             buffer.append(data[data_index])
#             data_index = (data_index + 1) % len(data)
#         self.got.set_value('data_index', data_index)
#         return batch, labels
#
#     def generate_batch_cbow(self, data, batch_size, num_skips, skip_window):
#         '''
#         Batch generator for CBOW (Continuous Bag of Words).
#         batch should be a shape of (batch_size, num_skips)
#
#         Parameters
#         ----------
#         data: list of index of words
#         batch_size: number of words in each mini-batch
#         num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
#         skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
#         '''
#         # global data_index
#         data_index = self.got.get_value('data_index', -1)
#         assert batch_size % num_skips == 0
#         assert num_skips <= 2 * skip_window
#         batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
#         labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
#         span = 2 * skip_window + 1  # [ skip_window target skip_window ]
#         buffer = collections.deque(maxlen=span)  # used for collecting data[data_index] in the sliding window
#         # collect the first window of words
#         for _ in range(span):
#             buffer.append(data[data_index])
#             data_index = (data_index + 1) % len(data)
#         # move the sliding window
#         for i in range(batch_size):
#             mask = [1] * span
#             mask[skip_window] = 0
#             batch[i, :] = list(compress(buffer, mask))  # all surrounding words
#             labels[i, 0] = buffer[skip_window]  # the word at the center
#             buffer.append(data[data_index])
#             data_index = (data_index + 1) % len(data)
#         self.got.set_value('data_index', data_index)
#         return batch, labels
