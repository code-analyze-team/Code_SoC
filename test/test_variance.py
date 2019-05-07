from build_data.dot_parse import dot_parser
from graphs.graph import Graph
from build_data.embedding_ops import emb_ops
from build_data.global_values import global_value
from build_data.build_word2vec_data import w2v_builder
from metrics.variance import metric_variance
from graphs.graph import find_label
import tensorflow as tf
import os
import psutil
import sys

path = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo/'

parser = dot_parser(path)
parser.read_dot_files()
parser.map_method_dot()
parser.map_dot_method()
parser.get_relabelled_graphs()

# init embedding op class
emb_op = emb_ops('/home/qwe/zfy_courses/w2v_learn/log/embeddings.tsv')
got = global_value()
builder = w2v_builder()
metric = metric_variance()
label2id = got.get_label_tsvid_map()

errcount = 0
# g_var_map = {}
# g_nvar_map = {}
with tf.Session() as sess:
    count = 0
    for g in parser.relabelled_graph_list:
        graph = Graph(g)
        embs = []
        for node in graph.nodes():
            label = graph.get_label(node)
            # label = find_label(g, node)
            label = builder.preprocess(label)
            if label2id.get(label) == None:
                errcount += 1
                id = 0
            else:
                id = label2id[label]
            emb = emb_op.lookup_single(id)
            embs.append(emb)
        print('now computing: ', count)
        matrix = metric.get_compute_matrix(embs)
        print(matrix)
        var = metric.compute_variance(matrix)
        # nvar = metric.compute_standard_variance(matrix)
        # g_var_map.setdefault(g,None)
        # g_var_map[g] = var.eval()
        # g_nvar_map.setdefault(g, None)
        # g_nvar_map[g] = nvar.eval()
        count += 1
print('err: ' + str(errcount))
