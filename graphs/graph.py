# -*- coding: utf-8 -*-

"""
a class with graph utils
level: method (correspond to a .dot file)
dot version: use abstracted jimple representation; all call relations (in project + jdk + 3rd party) inside the dot file
graph version: use int number (0, 1, 2 ...) as node id, different node id may contains same label.
"""

import logging
import sys
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict, Iterable
import random
from random import shuffle
from itertools import product,permutations
from scipy.io import loadmat
from scipy.sparse import issparse


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
        return self.graph.in_edges(node)


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
        # TODO
        return None

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