"""
random walks --> training batches --> nl tokens --> label embedding map
input: dot files
output:
        1. random walks file
        2. training batches (on jupyter for now)
        3. nl words file
        4. embedding map file
note that preprocess for nl shuold only in method label2tokens(), not in any other places.
also, preprocess for label shuold only be used in method generate_random_walks(), not other places.
"""
from build_data.dot_parse import dot_parser
from graphs.graph import Graph
from build_data.nl_transform import nl_transformer
from build_data.embedding_ops import emb_ops
from build_data.preprocess import preprocessor
from itertools import compress
import collections
import numpy as np
import tensorflow as tf
import gc

def generate_random_walks(dot_path, data_path, walk_mode):
    parser = dot_parser(dot_path)
    parser.read_dot_files()
    parser.map_method_dot()
    parser.map_dot_method()
    parser.get_relabelled_graphs()
    graphs = parser.relabelled_graph_list

    random_walks = []
    for g in graphs:
        graph = Graph(g)
        l = len(g.nodes())
        walk_length = int(l / 2 + 1)
        walk_times = 8 * l
        if walk_mode == 'origin':
            walk_method = graph.random_walk_adjacent
        elif walk_mode == 'in_graph':
            walk_method = graph.random_walk_graph
        elif walk_mode == 'with_edge':
            walk_method = graph.random_walk_graph_edge_sampled
        for i in range(walk_times):
            walk = walk_method(walk_length)
            random_walks.append(walk)

    labels = []
    with open(data_path, 'w') as file:
        for walk in random_walks:
            for word in walk:
                word = preprocessor.clean_label(word)
                file.writelines(word + '#')
                labels.append(word)


    graph_labels = []
    for g in graphs:
        graph = Graph(g)
        for node in graph.nodes():
            graph_labels.append(graph.get_label(node))

    print(len(set(labels)), len(set(graph_labels)))
    # assert len(set(labels)) == len(set(graph_labels)) + 4
    return None

def generate_batches_cbow(data_path, batch_path, batch_size=64, skip_window=1):
    """
    generate a training batch.
    """

    '''read data.'''
    f = open(data_path, 'r')
    data = f.read()
    words = data.split('#')
    f.close()

    '''Process raw inputs into a dataset.'''
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common())
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


    '''generate training batches.'''
    global data_index #TODO
    num_skips = 2 * skip_window
    assert batch_size % num_skips == 0
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # used for collecting data[data_index] in the sliding window
    # collect the first window of words
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # move the sliding window
    for i in range(batch_size):
        mask = [1] * span
        mask[skip_window] = 0
        batch[i, :] = list(compress(buffer, mask))  # all surrounding words (line i in batch)
        labels[i, 0] = buffer[skip_window]  # the word at the center
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def generate_nl_tokens(data_path, nl_path):
    trans = nl_transformer()
    with open(data_path, 'r') as f:
        with open(nl_path, 'w') as f2:
            text = f.read()
            labels = text.split('#')
            for label in labels:
                nl = trans.label2tokens(label)
                print(label, nl)
                for t in nl:
                    f2.write(t + ' ')
    return None


def generate_embedding_map(nl_tsv, embeddings_tsv, label_path, write_path):
    op = emb_ops(embeddings_tsv)
    trans = nl_transformer()

    '''get nl and corresponding id in metadata.tsv'''
    nl_id_map = {}
    with open(nl_tsv, 'r') as f:
        count = 0
        for line in f.readlines():
            label = line.strip('\n')
            id_ = count
            nl_id_map.setdefault(label, '')
            nl_id_map[label] = id_
            count += 1

    '''get label set'''
    labels = []
    with open(label_path, 'r') as f:
        text = f.read().split('#')
        for label in text:
            labels.append(label)
    label_set = set(labels)

    with open(write_path, 'w') as f:
        with tf.Session() as sess:
            i = 0
            for label in label_set:
                print(i)
                tokens = trans.label2tokens(label)
                tokens_id = [nl_id_map.get(t) for t in tokens]
                embs = op.lookup_1d(tokens_id)
                mean = tf.reduce_mean(embs, axis=0)
                mean_value = sess.run(mean)
                for v in mean_value:
                    f.write(str(v) + ' ')
                f.write('\n')
                gc.collect()
                i += 1


def id_test():
    """
    1. random walk coverage
    2. data_index in batch generation
    3. id in data & id in metadata.tsv
    4. preprocess
    """

    return None

dot_path = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo'
data_path = '/home/qwe/zfy_data/SoC/data_new/random_walks#with_edge#union'
nl_path = '/home/qwe/zfy_data/SoC/data_new/nl#with_edge#union'
# generate_random_walks(dot_path, data_path, 'with_edge')
print('pycharm is shit')
generate_nl_tokens(data_path, nl_path)