from graphs.graph import Graph

from build_data.dot_parse import dot_parser

import numpy as np


path = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo/'
write_path = '/home/qwe/zfy_lab/pdg2vec/data/random_walk_data_adjacent.txt'

parser = dot_parser(path)
parser.read_dot_files()
parser.map_method_dot()
parser.map_dot_method()
parser.get_relabelled_graphs()

graphs = parser.relabelled_graph_list


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
        walk = graph.random_walk_adjacent(walk_length)
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

