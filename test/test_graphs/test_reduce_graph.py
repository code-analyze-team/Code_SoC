from graphs.graph import Graph

from build_data.dot_parse import dot_parser
import networkx as nx
import matplotlib.pyplot as plt

path = '/home/qwe/zfy_lab/project_dots/spring-master/AST_CFG_PDGdotInfo/'

parser = dot_parser(path)
parser.read_dot_files()
parser.map_method_dot()
parser.map_dot_method()
parser.get_relabelled_graphs_partial(10)

graphs = parser.relabelled_graph_list_partial

g1 = graphs[9]
graph = Graph(g1)

plt.subplot(211)
nx.draw(g1,with_labels=True, font_weight='bold')
# plt.show()
# plt.savefig('/home/qwe/zfy_lab/SoC/test/test_graphs/g1.png')

print('-------------------------------------------')
g2 = graph.reduce_graph_by_line_num()
for e in g2.edges():
    print(e)

for n in g2.nodes():
    print(n)
nx.draw(g2,with_labels=True, font_weight='bold')
# plt.show()
# plt.savefig('/home/qwe/zfy_lab/SoC/test/test_graphs/g2.png')