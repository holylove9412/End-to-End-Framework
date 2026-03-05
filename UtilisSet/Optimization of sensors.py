import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from swmm_api.input_file import read_inp_file
from torch_geometric.data import Data
from torch_geometric.nn import NNConv
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
current_file = Path(__file__)
current_dir = current_file.parent.parent

inp = read_inp_file(current_dir/'data'/'SWMM_data'/'test'/'networks'/"test.inp")

coordinates = inp.COORDINATES
junctions_df = inp.JUNCTIONS.frame
conduits_df = inp.CONDUITS.frame
xsections_df = inp.XSECTIONS.frame

conduits_df = conduits_df.merge(xsections_df[['shape', 'height']], left_index=True, right_index=True)

node_ids = junctions_df.index.tolist()
node_id_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

edges = []
edge_attrs = []

for idx, row in conduits_df.iterrows():
    from_node, to_node = row['from_node'], row['to_node']
    if from_node not in node_id_to_idx or to_node not in node_id_to_idx:
        continue
    from_idx = node_id_to_idx[from_node]
    to_idx = node_id_to_idx[to_node]
    diameter = row['height']  # 管径
    length = row['length']
    z1 = junctions_df.loc[from_node]['elevation']
    z2 = junctions_df.loc[to_node]['elevation']
    slope = max((z1 - z2) / length, 0.0001)
    capacity_score = (diameter ** 2) / length * slope

    edges.append((from_idx, to_idx))
    edge_attrs.append([diameter, length, slope, capacity_score])

from collections import defaultdict, deque

# 初始化特征容器
node_features = []
num_nodes = len(node_ids)

# 辅助结构
neighbor_diameters = defaultdict(list)
neighbor_slopes = defaultdict(list)
degree_in = defaultdict(int)
degree_out = defaultdict(int)
adj_reverse = defaultdict(list)

# 遍历边提取属性
for (from_idx, to_idx), (diameter, length, slope, capacity_score) in zip(edges, edge_attrs):
    neighbor_diameters[from_idx].append(diameter)
    neighbor_diameters[to_idx].append(diameter)
    neighbor_slopes[from_idx].append(slope)
    neighbor_slopes[to_idx].append(slope)
    degree_out[from_idx] += 1
    degree_in[to_idx] += 1
    adj_reverse[to_idx].append(from_idx)

def compute_upstream_distances(num_nodes, adj_reverse):
    distances = [0.0] * num_nodes
    visited = [False] * num_nodes
    queue = deque()

    source_nodes = [i for i in range(num_nodes) if degree_in[i] == 0]
    for src in source_nodes:
        queue.append((src, 0.0))
        visited[src] = True

    while queue:
        current, dist = queue.popleft()
        distances[current] = max(distances[current], dist)
        for upstream in adj_reverse[current]:
            if not visited[upstream]:
                queue.append((upstream, dist + 1.0))  # 每条边算一步
                visited[upstream] = True

    return distances

upstream_distances = compute_upstream_distances(num_nodes, adj_reverse)

for node_id in node_ids:
    idx = node_id_to_idx[node_id]
    elevation = junctions_df.loc[node_id]['elevation']
    degree = degree_in[idx] + degree_out[idx]
    avg_diameter = sum(neighbor_diameters[idx]) / len(neighbor_diameters[idx]) if neighbor_diameters[idx] else 0.0
    avg_slope = sum(neighbor_slopes[idx]) / len(neighbor_slopes[idx]) if neighbor_slopes[idx] else 0.0
    upstream_dist = upstream_distances[idx]
    source_sink_flag = 1.0 if degree_in[idx] == 0 else (-1.0 if degree_out[idx] == 0 else 0.0)

    node_features.append([
        elevation,
        degree,
        avg_diameter,
        avg_slope,
        upstream_dist,
        source_sink_flag
    ])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
num_nodes = len(node_ids)
x = torch.tensor(node_features, dtype=torch.float)
node_features_numpy = np.array(node_features)

num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(node_features_numpy)

cluster_df = pd.DataFrame({
    'Node_Index': range(num_nodes),
    'Node_ID': node_ids,
    'Cluster_Label': cluster_labels
})

print(cluster_df)
cluster_df.to_csv("cluster_results.csv", index=False)
print(" 聚类完成，结果已保存为 cluster_results.csv")

