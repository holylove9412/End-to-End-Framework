import torch
import torch as th
import copy
import os,time
from pathlib import Path
import zarr
from torch_geometric.utils import from_networkx
import networkx as nx
import numpy as np
import pickle
from torch_geometric.data import Data
import matplotlib.cm as cm
import matplotlib.colors as colors
from swmm_api import read_inp_file
from torch_geometric.utils import add_self_loops
from swmm_api.input_file.sections import FilesSection,Control
from swmm_api.input_file.section_lists import NODE_SECTIONS,LINK_SECTIONS

class Neighbour_nodes():

    def __init__(self):
        self.inp_dir = r'D:\pythonProject4\FloodingRiskAssessment\data\SWMM_data\jiaotongnan\networks\jiaotongnan.inp'
        self.inp = read_inp_file(self.inp_dir)
        self.nodes = list(self.inp.COORDINATES)
        self.links = [link for label in LINK_SECTIONS if label in self.inp for link in getattr(self.inp,label).values()]
        self.edge_list = self.get_edge_list()

    def edge_us_with_depth(self, G, start_node, max_depth):
        visited = set()
        edge_us = []
        # edge_weights = []
        stack = [(start_node, 0)]  # (current_node, current_depth)
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                continue
            visited.add(node)
            for pred in G.predecessors(node):
                edge_us.append((pred, node, depth))
                if pred not in visited:
                    stack.append((pred, depth + 1))
        return edge_us

    def edge_ds_with_depth(self, G, start_node, max_depth):
        visited = set()
        edge_ds = []
        # edge_weights = []
        stack = [(start_node, 0)]  # (current_node, current_depth)
        while stack:
            node, depth = stack.pop()
            if depth > max_depth:
                continue
            visited.add(node)
            for succ in G.successors(node):
                edge_ds.append((node, succ, depth))
                if succ not in visited:
                    stack.append((succ, depth + 1))
        return edge_ds


    def get_edge_list(self):
        edges, lengths = [], []
        for link in self.links:
            if link.from_node in self.nodes and link.to_node in self.nodes:  # 两个属性由大写变为小写FromNode、ToNode
                edges.append((self.nodes.index(link.from_node), self.nodes.index(link.to_node)))  ##获取边节点的索引
                lengths.append(getattr(link, 'length', 0.0))

        return np.array(edges), np.array(lengths)

    def get_neighbour_nodes(self,n,find_depth=1):
        edges = self.edge_list[0]
        X = nx.MultiDiGraph()

        for u, v in edges:
            X.add_edge(u, v)


        edges_up, edges_ds = self.edge_us_with_depth(X, n, find_depth), self.edge_ds_with_depth(X, n,find_depth)

        return edges_up, edges_ds
    def get_G(self):
        edges = self.edge_list[0]
        X = nx.MultiDiGraph()

        for u, v in edges:
            X.add_edge(u, v)
        return X