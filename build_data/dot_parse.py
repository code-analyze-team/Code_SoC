# -*- coding: utf-8 -*-

"""
parse dot files , generate relabelled graphs in scatter mode
graph_level: method
dot_version: use abstracted jimple representation; all call relations (in project + jdk + 3rd party) inside the dot file
graph_version: use int number (0, 1, 2 ...) as node id, different node id may contains same label.
"""
import logging
import os
import networkx as nx
from networkx.readwrite.gexf import read_gexf, write_gexf
import pymysql
from networkx.drawing.nx_pydot import write_dot,read_dot
import pydot
import re
from build_data.preprocess import preprocessor
from graphs.graph import Graph


logger = logging.getLogger('cg2vec')

__author__ = 'zfy'

LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class dot_parser:
    """parse dots , get graphs from the dots, and relabel them"""

    def __init__(self, paths):
        self.path = paths.split('#')
        self.cfg_maps = []
        self.cfg_dot_files = []
        self.third_party_maps = []
        self.third_party_dot_files = []
        self.md_dot_map = {}
        self.dot_md_map = {}
        self.relabelled_graph_list = []
        self.relabelled_graph_list_partial = []
        self.reduced_graphs = []

    def read_dot_files(self):
        """
        read dot files, count on cfg files, cfg_map files, 3rd_party files, 3rd_map files
        3rd_party are deprecated now, because we dont consider 3rd method as a dot file any more.
        :return:
        """
        cfg_maps = []
        cfg_dot_files = []
        for path in self.path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file == 'cfg_map.txt':
                        fullname = os.path.join(root, file)
                        parent = root
                        cfg_maps.append((fullname, parent))
                    if file.endswith('CFG.dot'):
                        fullname = os.path.join(root, file)
                        cfg_dot_files.append(fullname)


        self.cfg_maps = cfg_maps
        self.cfg_dot_files = cfg_dot_files

    def map_method_dot(self):
        """
        get a map with key: method_name && value: dot file name
        :return: the map
        """
        cfg_maps = self.cfg_maps
        md_dot_map = {}
        for map in cfg_maps:
            fullname = map[0]
            parent = map[1]
            with open(fullname, 'r') as file:
                content = file.readlines()
                for line in content:
                    line = line.split(':')
                    dot_num = line[0]
                    md_name = line[1].strip()
                    full_dotfile_name = parent + '/' + dot_num + '_CFG.dot'
                    md_dot_map.setdefault(md_name, [])
                    md_dot_map[md_name].append(full_dotfile_name)
        self.md_dot_map = md_dot_map
        return md_dot_map

    def map_dot_method(self):
        """
        get a map wtih key: dot file name && value: method name
        :return: the map
        """
        cfg_maps = self.cfg_maps
        dot_md_map = {}
        for map in cfg_maps:
            fullname = map[0]
            parent = map[1]
            with open(fullname, 'r') as file:
                content = file.readlines()
                for line in content:
                    line = line.split(':')
                    dot_num = line[0]
                    md_name = line[1].strip()
                    #             print(md_name)
                    full_dotfile_name = parent + '/' + dot_num + '_CFG.dot'
                    #             if (md_name.find('<init>') == -1):
                    dot_md_map.setdefault(full_dotfile_name, [])
                    dot_md_map[full_dotfile_name].append(md_name)
        self.dot_md_map = dot_md_map
        return dot_md_map

    def get_relabelled_graphs(self):
        """
        dots --> networkx graphs
        relabelling:
        1. graph level: graph name / dot path
        2. node level: node name(1 -> G1/1)/ label(jimple represetation) / dot path(same as graph level) / count (first, n , last)
        3. edge level: label(cfg, dataflow, extra, calling)
        :return: relabelled graph list
        """
        count = 0
        errcount = 0
        g_list = []
        for dot_file in self.cfg_dot_files:
            print('now at: ', count)
            count += 1
            try:
                g = read_dot(dot_file)
                g.graph['name'] = 'G'
                g.graph['name'] += str(count)
                g.graph['id'] = dot_file
                node_count = 0
                for node in g.nodes():
                    g.node[node].setdefault('name', '')
                    g.node[node]['name'] = g.graph['name'] + '/' + node
                    g.node[node]['from'] = g.graph['id']
                    g.node[node]['count'] = node_count
                    node_count += 1
                for node in g.nodes():
                    if g.nodes[node]['count'] == 0:
                        g.nodes[node]['count'] = 'first'
                    if g.nodes[node]['count'] == len(g.nodes()) - 2:
                        g.nodes[node]['count'] = 'last'
                    if g.nodes[node]['count'] == len(g.nodes()) - 1:
                        g.nodes[node]['count'] = 'method'
                edge_count = 0
                for e in g.edges:
                    g.edges[e]['id'] = edge_count
                    labels = g.edges[e]
                    if labels.get('color') == None:
                        g.edges[e]['label'] = 'cfg'
                    elif labels.get('color') == 'red':
                        g.edges[e]['label'] = 'data_flow'
                    elif labels.get('color') == 'blue':
                        g.edges[e]['label'] = 'extra'
                    elif labels.get('color') == 'green':
                        g.edges[e]['label'] = 'calling'
                    else:
                        print('err')
                    edge_count += 1
                g_list.append(g)
            except Exception as e:
                errcount += 1
                print(e)
                print(e.with_traceback)

        # print('well done')
        # print(errcount)
        self.relabelled_graph_list = g_list
        return g_list

    # python dose not support function overload
    def get_relabelled_graphs_partial(self, num):
        """
        for case dont need full glist
        :param num: expect number
        :return: partial list of relabelled graphs
        """
        count = 0
        errcount = 0
        g_list = []
        for dot_file in self.cfg_dot_files[:num]:
            count += 1
            try:
                g = read_dot(dot_file)
                g.graph['name'] = 'G'
                g.graph['name'] += str(count)
                g.graph['id'] = dot_file
                node_count = 0
                for node in g.nodes():
                    g.node[node].setdefault('name', '')
                    g.node[node]['name'] = g.graph['name'] + '/' + node
                    g.node[node]['from'] = g.graph['id']
                    g.node[node]['count'] = node_count
                    node_count += 1
                for node in g.nodes():
                    if g.nodes[node]['count'] == 0:
                        g.nodes[node]['count'] = 'first'
                    if g.nodes[node]['count'] == len(g.nodes()) - 2:
                        g.nodes[node]['count'] = 'last'
                    if g.nodes[node]['count'] == len(g.nodes()) - 1:
                        g.nodes[node]['count'] = 'method'
                edge_count = 0
                for e in g.edges:
                    g.edges[e]['id'] = edge_count
                    labels = g.edges[e]
                    if labels.get('color') == None:
                        g.edges[e]['label'] = 'cfg'
                    elif labels.get('color') == 'red':
                        g.edges[e]['label'] = 'data_flow'
                    elif labels.get('color') == 'blue':
                        g.edges[e]['label'] = 'extra'
                    elif labels.get('color') == 'green':
                        g.edges[e]['label'] = 'calling'
                    else:
                        print('err')
                    edge_count += 1
                g_list.append(g)
            except Exception as e:
                errcount += 1
                print(e)
                print(e.with_traceback)

        # print('well done')
        # print(errcount)
        self.relabelled_graph_list_partial = g_list

    def get_reduced_graphs(self):
        reduced_graphs = []
        for g in self.relabelled_graph_list:
            graph = Graph(g)
            g_ = graph.reduce_graph_by_line_num()
            reduced_graphs.append(g_)

        self.reduced_graphs = reduced_graphs
        return reduced_graphs

    # def get_all_labels_jimple(self):
    #     """
    #     since not all labels are in jimple representation
    #     this method gather all the label in jimple form from all the dot files
    #     to elimate repetive jimples, we return them in a set
    #     :return: a set of all jimple representations
    #     """
    #     """
    #     note that all the move that requires a full satck of jimples shuold use this method
    #     and preprocess shuold be done
    #     """
    #     jimples = []
    #     for g in self.relabelled_graph_list:
    #         for node in g.nodes():
    #             label = str(g.nodes[node]['label'])
    #             label = preprocessor.clean_word(label)
    #             if label.find('Stmt') != -1:
    #                 jimples.append(label)
    #     jimple_set = set(jimples)
    #     self.jimples = jimples
    #     self.jimple_set = jimple_set
    #     return jimple_set