import os
import argparse
import sys

from build_data.build_word2vec_data import w2v_builder
from shallow_models.basic_word2vec import Word2Vec

# data_index = 0


builder = w2v_builder()
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.')
flags, unused_flags = parser.parse_known_args()


log_dir = '/home/qwe/PycharmProjects/cg2vec/res/log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


filename = '/home/qwe/PycharmProjects/cg2vec/data/data.txt'
vocabulary = builder.read_local_data(filename)
print('Data size', len(vocabulary))


# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 2495

data, count, unused_dictionary, reverse_dictionary = builder.build_dataset(
      vocabulary, vocabulary_size)
del vocabulary  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])




batch, labels = builder.generate_batch_cbow(data, batch_size=8, num_skips=2, skip_window=1)
print(batch)

w2v = Word2Vec(batch_size=128, num_skips=2, skip_window=1, architecture='cbow', embedding_size=128,
              vocabulary_size=50000, loss_type='nce_loss', n_neg_samples=64, optimize='Adagrad',
              learning_rate=1.0, n_steps=100001, valid_size=16, valid_window=100)

w2v.fit(data)