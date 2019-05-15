from graphs.graph import Graph

from build_data.dot_parse import dot_parser

import numpy as np


path1 = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo/'
path2 = '/home/qwe/zfy_lab/project_dots/1.Spring core/AST_CFG_PDGdotInfo/'
write_path = '/home/qwe/zfy_lab/pdg2vec/data/all_proj/random_walk_data_edge#all_proj.txt'

parser1 = dot_parser(path1)
parser1.read_dot_files()
parser1.map_method_dot()
parser1.map_dot_method()
parser1.get_relabelled_graphs()

graphs_1 = parser1.relabelled_graph_list

parser2 = dot_parser(path2)
parser2.read_dot_files()
parser2.map_method_dot()
parser2.map_dot_method()
parser2.get_relabelled_graphs()

graphs_2 = parser2.relabelled_graph_list

graphs = graphs_1 + graphs_2

print(len(graphs), len(graphs_1), len(graphs_2))

"""
to define a suitable walk length
we print mean, median, outnum of lengths of graphs
as th result suggests, lengths of graphs varies, so we determined to use variable walk length
walk length = len(g.nodes()) / 2 + 1

"""

# lengths = []
# for g in graphs:
#     l = len(g.nodes())
#     lengths.append(l)
# mean = np.mean(lengths)
# middle = np.median(lengths)
# counts = np.bincount(lengths)
# outnum = np.argmax(counts)
# print(mean, middle, outnum)

random_walks = []
random_walks_label = []
label_nl_map = {}
label_id_map = {}

"""
start random walk
for each graph, walk length = len(g.nodes()) / 2 + 1
for each graph, walk times = len(g.nodes())
"""
for g in graphs:
    graph = Graph(g)
    print(g.graph['id'])
    l = len(g.nodes())
    walk_length = int(l/2 + 1)
    walk_times = 8 * l
    for i in range(walk_times):
        walk = graph.random_walk_graph_edge_sampled(walk_length)
        random_walks.append(walk)
        print(walk)
print(len(random_walks))

labels_in_walk = []
# write to disk
with open(write_path, 'w') as file:
    for walk in random_walks:
        for word in walk:
            file.writelines(word + '#')
            labels_in_walk.append(word)

lset= set(labels_in_walk)

labels_origin = []
for g in graphs:
    graph = Graph(g)
    for node in graph.nodes():
        labels_origin.append(graph.get_label(node))

lset_ = set(labels_origin)

#5095 5105
print(len(lset), len(lset_))

