# -*- coding: utf-8 -*-

"""
a class with graph utils
level: method (correspond to a .dot file)
dot version: use abstracted jimple representation; all call relations (in project + jdk + 3rd party) inside the dot file
graph version: use int number (0, 1, 2 ...) as node id, different node id may contains same label.
"""

import logging
import random
import networkx as nx
import matplotlib.pyplot as plt
from build_data.preprocess import preprocessor


logger = logging.getLogger('cg2vec')

__author__ = 'zfy'

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Graph:

    """
    basic implementation & extension of networkx.MultiDiGraph()
    """

    def __init__(self, graph):
        """
        receive a networkx.graph after relabelling from dot files
        use int number as node id
        :param graph: a relabelled networkx.graph()
        """
        self.graph = graph

    def nodes(self):
        """return all node id of the graph"""
        return list(self.graph.nodes())


    def edges(self):
        """return all edges of a graph"""
        return list(self.graph.edges())


    def get_label(self, node):
        """return the label of a node"""
        return self.graph.nodes[node]['label']

    def get_nl(self, node):
        """return natural language of a node"""
        return self.graph.nodes[node]['natural_language']

    def get_lineNum(self,node):
        """return line number of a node in the graph"""
        if self.graph.nodes[node].get('lineNum') != None:
            return self.graph.nodes[node]['lineNum']
        else:
            return -1

    def get_name(self, node):
        """return name of a node"""
        return self.graph.nodes[node]['name']



    def adjacency_iter(self):
        """
        get the adjacent list of a multi-directed-graph
        note that if there are multi lines between two nodes, only one is recorded in a adj_list
        e.g. (0, 1, color='red) and (0, 1, color='blue') --> 0: 1
        TODO: will multi-lines affect random walk?
        :return: dict -> {node : its adj_list}
        """
        adj_dict = {}
        for node, neibour in self.graph.adjacency():
            adj_dict.setdefault(node, [])
            for v in neibour:
                adj_dict[node].append(v)
        return adj_dict


    def subgraph(self):
        #TODO
        return None


    def make_undirected(self):
        #TODO
        return None


    def make_consistent(self):
        #TODO
        return None


    def remove_self_loop(self):
        """
        remove self-loop in graph
        e.g. (1 -> 1)
        :return refined adjacent dict
        """

        removed_num = 0

        adjacent = self.adjacency_iter()
        for node, neibour in adjacent:
            if node in neibour:
                neibour.remove(node)
                removed_num += 1

        logger.info('remove self-loops: removed {} self loops'.format(removed_num))
        return adjacent


    def check_self_loop(self):
        """
        check if a graph has any self-loop
        :return: true or false
        """

        adjacent = self.adjacency_iter()
        for node, neibour in adjacent:
            if node in neibour:
                return True

        return False


    def has_edge(self):
        #TODO
        return None


    def degree(self):
        #TODO
        return None


    def order(self):
        #TODO
        return None


    def number_of_edges(self):
        #TODO
        return None


    def number_of_nodes(self):
        #TODO
        return None


    def out_edges(self, node):
        """return all out edges of a node"""
        return list(self.graph.out_edges(node))


    def in_edges(self, node):
        """return all out edges of a node"""
        return list(self.graph.in_edges(node))


    def out_edges_multi(self, node):
        """return all out edges in 3-tuple form"""
        out_edges_3_form = []
        out_edges = self.out_edges(node)
        for edge in out_edges:
            s = edge[0]
            t = edge[1]
            multi = self.graph[s][t]
            for i in range(len(multi)):
                label = self.graph[s][t][i]['label']
                out_edges_3_form.append((s,t,label))
        return out_edges_3_form


    def in_edges_multi(self, node):
        """return all in edges in 3-tuple form"""
        in_edges_3_form = []
        in_edges = self.in_edges(node)
        for edge in in_edges:
            s = edge[0]
            t = edge[1]
            multi = self.graph[s][t]
            for i in range(len(multi)):
                # print('i: ' + str(i))
                # print(s, t)
                label = self.graph[s][t][i]['label']
                in_edges_3_form.append((s, t, label))
        return in_edges_3_form

    def find_tail_node(self, head, edge):
        """
        given a head node and a edge, find the tail node
        :param head: head node
        :param edge: edge of 3-tuple form: (u, v, k='id')
        :return: tail node
        """
        s = edge[0]
        t = edge[1]
        return t

    def draw(self):
        plt.subplot(221)
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    @staticmethod
    def draw_save(graph, path):
        plt.subplot(211)
        nx.draw(graph, with_labels=True, font_weight='bold')
        # plt.show()
        plt.savefig(path)



    def random_walk_adjacent(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        returns a truncated random walk.
        :param path_length: length of a random walk
        :param alpha: probability of restart
        :param rand: random seed
        :param start: the start node of random walk
        :return: a generated random walk
        """
        if start:
            path = [start]
        else:
            # sampling nodes not edges
            path = [rand.choice(self.nodes())]

        while (len(path) < path_length):
            cur = path[-1]
            cur_neibour = self.adjacency_iter().get(cur)
            if len(cur_neibour) > 0:
                if rand.random() > alpha:
                    path.append(rand.choice(cur_neibour))
                else:
                    path.append(path[0])
            else:
                # this is how we truncate the walk
                break

        return [self.get_label(node) for node in path]


    def random_walk_graph(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        returns a truncated random walk that use original graph
        :param path_length: length of a random walk
        :param alpha: probability of restart
        :param rand: random seed
        :param start: the start node of random walk
        :return: a generated random walk
        """

        if start:
            path = [start]
        else:
            # sample a start node
            # sampling edges not nodes
            path = [rand.choice(self.nodes())]

        while (len(path) < path_length):
            cur = path[-1]
            out_edges = self.out_edges_multi(cur)
            if len(out_edges) > 0:
                if rand.random() > alpha:
                    selected_edge = rand.choice(out_edges)
                    next_node = self.find_tail_node(cur, selected_edge)
                    path.append(next_node)
                else:
                    path.append(path[0])
            else:
                break
        return [self.get_label(node) for node in path] # node, graph, dot_name


    def random_walk_graph_edge_sampled(self, path_length, alpha=0, rand=random.Random(), start=None):
        """
        returns a truncated random walk that use original graph, edges are sampled in walks
        :param path_length: length of a random walk
        :param alpha: probability of restart
        :param rand: random seed
        :param start: the start node of random walk
        :return: a generated random walk with edge sampled
        """

        if start:
            path = [start]
        else:
            # sample a start node
            # sampling edges not nodes
            path = [rand.choice(self.nodes())]

        edge_path = []
        while (len(path) < path_length):
            cur = path[-1]
            out_edges = self.out_edges_multi(cur)
            if len(out_edges) > 0:
                if rand.random() > alpha:
                    selected_edge = rand.choice(out_edges)
                    edge_path.append(selected_edge)
                    next_node = self.find_tail_node(cur, selected_edge)
                    path.append(next_node)
                else:
                    path.append(path[0])
            else:
                break
        res = []
        for i in range(0, len(path)-1):
            # print(self.get_label(path[i]))
            # print(edge_path[i][2])
            res.append(self.get_label(path[i]))
            res.append(edge_path[i][2])
        res.append(self.get_label(path[len(path)-1]))
        return res

    def reduce_graph_by_line_num(self):
        """
        1. given a line number, find all corresponding nodes : node_set
        2. find all edges link with node in node_set : edges_set = in_set + out_set
        3. remove inner edges : inner_set
        4. add a new node : new_node = label1 + label2 + ...
        5. attach remaining edges with new_node : remain_set = edges_set - inner_set
        6. delete inner nodes : inner_set
        7. relabel
        :return: relabelled graph
        """

        map = {}
        reversed_map = {}
        for node in self.graph.nodes():
            name = node
            line_num = self.get_lineNum(node)

            map.setdefault(name, [])
            map[name].append(line_num)
            reversed_map.setdefault(line_num, [])
            reversed_map[line_num].append(name)
        assert len(map.keys()) == len(self.graph.nodes())
        print(reversed_map)

        for num in reversed_map.keys():
            if int(num) != -1:
                # step1 : node_set
                node_set = reversed_map.get(num)

                # step2 : edges_set
                in_set = []
                out_set = []
                for node in node_set:
                    for e in self.out_edges_multi(node):
                        out_set.append(e)
                    for e in self.in_edges_multi(node):
                        in_set.append(e)
                edges_set = in_set + out_set
                edges_set = sorted(set(edges_set), key=edges_set.index)

                # step3 : edges_set = edges_set - inner_set
                inner_edges = []
                inner_nodes = reversed_map.get(num)
                # print(num +': ' + ' '.join(inner_nodes))
                for e in edges_set:
                    if e[0] in inner_nodes and e[1] in inner_nodes:
                        edges_set.remove(e)
                        inner_edges.append(e)

                # step4 : add a new graph
                new_node = inner_nodes[0] + '_'
                new_label = ''
                for node in inner_nodes:
                    l = self.get_label(node)
                    l = preprocessor.clean_word(l)
                    new_label += l + '\t'
                new_label = new_label.strip('\t')
                new_linenum = num
                self.graph.add_node(new_node, label=new_label, lineNum=new_linenum)

                # step5 : add new edges
                for e in edges_set:
                    self.graph.remove_edge(e[0], e[1])
                    if e in in_set:
                        source = e[0]
                        target = new_node
                        label = e[2]
                        self.graph.add_edge(source, target, label=label)

                    if e in out_set:
                        source = new_node
                        target = e[1]
                        label = e[2]
                        self.graph.add_edge(source, target, label=label)

                # step6 : remove inner nodes and edges
                self.graph.remove_nodes_from(inner_nodes)
                # for e in inner_edges:
                #     self.graph.remove_edge(e[0], e[1], e[2])

        return self.graph




        # '''relabel the graph'''
        # node_count_map = {}
        # count_label_map = {}
        # count = 0
        # for node in self.graph.nodes():
        #     label = self.get_label(node)
        #     node_count_map.setdefault(node, '')
        #     node_count_map[node] = str(count)
        #     count_label_map.setdefault(str(count), '')
        #     count_label_map[str(count)] = label
        #     count +=1
        # relabelled_graph = nx.relabel_nodes(self.graph, node_count_map)
        # for node in relabelled_graph.nodes():
        #     label = count_label_map.get(node)
        #     relabelled_graph.nodes[node]['label'] = label
        #
        # relabelled_graph = Graph(relabelled_graph)
        # # for node in relabelled_graph.nodes():
        # #     print(node, relabelled_graph.get_label(node))
        #
        # return relabelled_graph




