import networkx as nx
g = nx.MultiDiGraph()

g.add_edge(1,2, weight=4.7)
g.add_edge(1,2, color='red')

for edge in g.edges():
    multi = g[1][2]
    print(len(g[1][2]))