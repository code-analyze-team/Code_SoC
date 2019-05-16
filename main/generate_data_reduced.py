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

def generate_random_walks_reduced(dot_path, data_path, walk_mode):
    parser = dot_parser(dot_path)
    parser.read_dot_files()
    parser.map_method_dot()
    parser.map_dot_method()
    parser.get_relabelled_graphs()
    parser.get_reduced_graphs()
    graphs = parser.reduced_graphs

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
                assert word != ''
                # print(word)
                file.write(word + '#')
                labels.append(word)


    graph_labels = []
    for g in graphs:
        graph = Graph(g)
        for node in graph.nodes():
            graph_labels.append(graph.get_label(node))

    print(len(set(labels)), len(set(graph_labels)))
    # assert len(set(labels)) == len(set(graph_labels)) + 4
    return None

def generate_nl_tokens(data_path, nl_path):
    trans = nl_transformer()
    with open(data_path, 'r') as f:
        with open(nl_path, 'w') as f2:
            text = f.read()
            labels = text.split('#')
            labels = labels[:-1]
            for label in labels:
                nl = trans.label2tokens(label)
                # print(label, nl)
                for t in nl:
                    f2.write(t + ' ')
    return None



def generate_embedding_map(glove_path, label_tsv, write_path):
    op = emb_ops(glove_path)
    trans = nl_transformer()

    '''get label set from meta_label.tsv'''
    labels = []
    with open(data_path, 'r') as f:
        text = f.read().split('#')
        for label in text:
            labels.append(label)

    print(len(labels))
    labels = labels[:-1]

    l2 = sorted(set(labels), key=labels.index)
    f = open(label_tsv, 'w')
    count = 0
    for label in l2:
        f.write(label + '\t' + str(count))
        f.write('\n')
        count += 1

    with open(write_path, 'w') as f:
        with tf.Session() as sess:
            i = 0
            for label in l2:
                print(i)
                tokens = trans.label2tokens(label)
                embs = op.lookup_glove_embedding_batch(tokens)
                mean = tf.reduce_mean(embs, axis=0)
                mean_value = sess.run(mean)
                for v in mean_value:
                    f.write(str(v) + ' ')
                f.write('\n')
                gc.collect()
                i += 1

dot_path = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo'
data_path = '/home/qwe/zfy_data/SoC/data_new/random_walks#with_edge#union#reduced'
nl_path = '/home/qwe/zfy_data/SoC/data_new/nl#with_edge#union#reduced'
tsv_path = '/home/qwe/zfy_data/SoC/data_new/label_mean_emb.tsv'
glove_path = '/home/qwe/zfy_data/SoC/data_new/nl_pre/glove/glove.6B.100d.txt'
label_tsv = '/home/qwe/zfy_data/SoC/data_new/labels.tsv'



# generate_random_walks_reduced(dot_path, data_path, 'with_edge')
# generate_nl_tokens(data_path, nl_path)
generate_embedding_map(glove_path, label_tsv, tsv_path)