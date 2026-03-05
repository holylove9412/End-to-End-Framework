import os
import pandas as pd
from swmm_api.input_file import read_inp_file
from torch_geometric.data import Data
from pathlib import Path
from scipy.sparse import csgraph
from collections import defaultdict, deque
from sklearn.metrics import silhouette_score
import networkx as nx
from UtilisSet.optimization_layout.graph_operator import *
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csgraph
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from UtilisSet.optimization_layout.data_loader import *
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from sklearn.cluster import KMeans
import os
import random
import numpy as np
import torch
import matplotlib.patches as mpatches
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 固定 CuDNN 后端行为（可选）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量（Python 内核）
    os.environ['PYTHONHASHSEED'] = str(seed)

# 用法
set_seed(42)  # 在一切随机操作前设置

def loss_function(pred, label):
    return F.binary_cross_entropy_with_logits(pred, label.float())
def mixed_negative_sampling(edge_index, num_nodes, num_neg_samples):
    device = edge_index.device
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    neg_samples = set()
    tries = 0
    while len(neg_samples) < num_neg_samples and tries < num_neg_samples * 10:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and adj[i, j] == 0:
            neg_samples.add((i, j))
        tries += 1
    neg_samples = torch.tensor(list(neg_samples), dtype=torch.long).T.to(device)
    return neg_samples
def get_Graph(node_ids,coordinates,links):
    G = nx.MultiDiGraph()
    for node_id in node_ids:
        G.add_node(node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']))
    for idx, row in links.items():
        from_node, to_node = idx[0], idx[1]
        G.add_edge(from_node, to_node)
    return G
def get_Graph_Idx(node_ids,coordinates,links):
    G = nx.MultiDiGraph()
    for node_idx,node_id in enumerate(node_ids):
        G.add_node(node_idx, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']))
    for idx, row in links.items():
        from_node, to_node = node_list.index(idx[0]), node_list.index(idx[1])
        G.add_edge(from_node, to_node)
    return G
def plt_nx_layout_without_monitored(node_ids,cluster_labels,clor):
    # 创建 NetworkX 图（无方向或可设为有向）
    import matplotlib.colors as colors
    import matplotlib.cm as cm

    G = nx.MultiDiGraph()
    for i, node_id in enumerate(node_ids):
        G.add_node(i, label=node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']),cluster=cluster_labels[i])
    for from_idx, to_idx in edges:
        G.add_edge(from_idx, to_idx)
    # 计算节点布局（图可视化布局）
    pos = nx.get_node_attributes(G, 'pos') # 或者 nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(10, 8))  # 创建 Figure 和 Axes
    x = np.array(cluster_labels)
    x_norm = (x - x.min()) / (x.max() - x.min())

    # 计算节点颜色范围
    norm = colors.Normalize(vmin=0, vmax=1)  # 归一化特征值范围
    sm = cm.ScalarMappable(cmap=clor, norm=norm)  # 绑定颜色映射
    sm.set_array([])  # 必须设置数组来激活 ScalarMappable

    # 绘制图
    nx.draw(G, pos, with_labels=False, node_size=120, font_size=8,node_color=x_norm, cmap=clor, ax=ax)

    # 获取颜色映射
    num_clusters = len(set(cluster_labels))

    nx.draw_networkx_edges(G, pos, arrows=False, arrowstyle='->', width=0.8, alpha=0.4)

    plt.title("clustring node layout", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    # 构建图例手柄：一个 cluster 一个颜色
    legend_elements = [
        mpatches.Patch(color=sm.to_rgba(i / max(num_clusters - 1, 1)), label=f'Cluster {i}')
        for i in range(num_clusters)
    ]

    # 显示图例
    plt.legend(
        handles=legend_elements,
        title="legend",
        loc="upper right",
        fontsize=10
    )
    plt.savefig(os.path.join(model_pt,'picture',f'alpha_{alpha}.png'))

    plt.show()
def plt_nx_layout(node_ids,cluster_labels,monitored_labels,clor):
    # 创建 NetworkX 图（无方向或可设为有向）
    import matplotlib.colors as colors
    import matplotlib.cm as cm

    G = nx.MultiDiGraph()
    for i, node_id in enumerate(node_ids):
        G.add_node(i, label=node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']),cluster=cluster_labels[i])
    for from_idx, to_idx in edges:
        G.add_edge(from_idx, to_idx)
    # 计算节点布局（图可视化布局）
    pos = nx.get_node_attributes(G, 'pos') # 或者 nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=(10, 8))  # 创建 Figure 和 Axes
    x = np.array(cluster_labels)
    x_norm = (x - x.min()) / (x.max() - x.min())

    # 计算节点颜色范围
    norm = colors.Normalize(vmin=0, vmax=1)  # 归一化特征值范围
    sm = cm.ScalarMappable(cmap=clor, norm=norm)  # 绑定颜色映射
    sm.set_array([])  # 必须设置数组来激活 ScalarMappable

    # 绘制图
    nx.draw(G, pos, with_labels=False, node_size=120, font_size=8,node_color=x_norm, cmap=clor, ax=ax)

    # 获取颜色映射

    num_clusters = len(set(cluster_labels))
    # colors = plt.cm.get_cmap("Pastel1", num_clusters)
    # node_colors = [colors(cluster_labels[i]) for i in range(len(node_ids))]
    #
    # # 绘制图形
    # plt.figure(figsize=(10, 8))
    # nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120, alpha=0.9)

    nx.draw_networkx_edges(G, pos, arrows=False, arrowstyle='->', width=0.8, alpha=0.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=monitored_labels,
        node_shape='*',  # 五角星形状
        node_color='lime',
        node_size=700,
        label='Monitored Node',
        edgecolors='black'
    )
    # nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Monitored node layout", fontsize=14)
    plt.axis("off")
    plt.tight_layout()

    # 构建图例手柄：一个 cluster 一个颜色
    legend_elements = [
        mpatches.Patch(color=sm.to_rgba(i / max(num_clusters - 1, 1)), label=f'Cluster {i}')
        for i in range(num_clusters)
    ]

    # 显示图例
    plt.legend(
        handles=legend_elements,
        title="legend",
        loc="upper right",
        fontsize=10
    )

    plt.show()

def sensor_covered_layout(G,sensor_node):

    pos = nx.get_node_attributes(G, 'pos')

    sensor_color = ['lime' for _ in range(len(sensor_node))]


    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_size=100, node_color='#d2e2ef', font_size=8, font_color='black')

    nx.draw_networkx_nodes(G, pos, nodelist=sensor_node, node_shape='*', edgecolors='black',node_size=700, node_color=sensor_color)
    plt.tight_layout()
    plt.show()

def edge_attr_extraction(links,modify_node_list,need_delet_nodes,node_elevaltion_attrs):
    node_id_map = {node_id: i for i, node_id in enumerate(modify_node_list)}
    edges = []
    edge_attrs = []
    for idx, row in links.items():
        from_node, to_node = idx[0], idx[1]
        if from_node in need_delet_nodes or to_node in need_delet_nodes:
            continue
        from_idx = node_id_map[from_node]
        to_idx = node_id_map[to_node]
        diameter = row['height']  # 管径
        length = row['length']
        slope_conduit = (row['in_offset']-row['out_offset'])/length
        z1 = node_elevaltion_attrs.loc[from_node]['elevation']
        z2 = node_elevaltion_attrs.loc[to_node]['elevation']
        # slope = max((z1 - z2) / length, 0.0001)
        slope = (z1 - z2) / length
        capacity_score = (diameter ** 2) / length * slope
        edges.append((from_idx, to_idx))
        edge_attrs.append([diameter, length, slope,capacity_score])
    return edges,edge_attrs
# 简化上游路径计算：用 BFS 估算从源节点到每个节点的最长路径长度
def compute_upstream_distances(num_nodes, adj_reverse,degree_in):
    distances = [0.0] * num_nodes
    visited = [False] * num_nodes
    queue = deque()

    # 将所有“入度为 0” 的节点视作源点
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


def node_attr_extraction(need_delet_nodes):
    node_features = []
    num_nodes = len(node_list)
    modify_node_list = [n for n in node_list if n not in need_delet_nodes]
    node_id_map = {node_id: i for i, node_id in enumerate(modify_node_list)}
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
        adj_reverse[to_idx].append(from_idx)  # 反向建图用于上游路径分析
    upstream_distances = compute_upstream_distances(num_nodes, adj_reverse,degree_in)

    # 构造节点特征向量
    for node_id in modify_node_list:
        idx = node_id_map[node_id]
        elevation = node_elevaltion_attrs.loc[node_id]['elevation']
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
    return node_features


def model_save(model,acc):
    if not os.path.exists(os.path.join(model_pt,model._get_name())):
        os.mkdir(os.path.join(model_pt,model._get_name()))
    if min(acc) == acc[-1]:
        print(f"bset_Loss: {acc[-1]:.4f}")
        torch.save(model, os.path.join(model_pt,model._get_name(),f'{model._get_name()}_best.pth'))
        return True
def edge_index_adj(edge_index):
    n_nodes = edge_index.max().item()+1
    adj = torch.zeros(n_nodes,n_nodes,dtype=torch.float32)
    adj[edge_index[0],edge_index[1]] = 1
    return adj


def elbow_method(z_emb, max_k=20):
    sse = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(z_emb)
        sse.append(kmeans.inertia_)  # SSE
    plt.plot(range(1, max_k + 1), sse, marker='o')
    plt.xlabel('Number of clusters k')
    plt.ylabel('SSE (Inertia)')
    plt.title('Elbow Method')
    plt.grid()
    plt.show()
def mixed_negative_sampling(edge_index, num_nodes, num_neg_samples):
    device = edge_index.device
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    neg_samples = set()
    tries = 0
    while len(neg_samples) < num_neg_samples and tries < num_neg_samples * 10:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and adj[i, j] == 0:
            neg_samples.add((i, j))
        tries += 1
    neg_samples = torch.tensor(list(neg_samples), dtype=torch.long).T.to(device)
    return neg_samples

def find_best_k_by_silhouette(z_emb, k_range=range(2, 20)):
    best_score = -1
    best_k = 2
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(z_emb)
        score = silhouette_score(z_emb, kmeans.labels_)
        print(f"k={k}, silhouette score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best k by silhouette: {best_k}")
    return best_k

def get_length_laplacian():
    N = len(modify_node_list)
    D = np.full((N, N), np.inf)
    for ix,value in enumerate(edges):
        i, j =value[0],value[1]
        length = edge_attrs[ix][1]
        D[i, j] = length
    # 设置高斯核带宽参数 σ
    sigma = np.nanmean(D[np.isfinite(D)])

    # 高斯变换
    A = np.exp(-(D ** 2) / (sigma ** 2))
    np.fill_diagonal(A, 0)
    A[~np.isfinite(D)] = 0  # 将 inf 变为 0
    D[~np.isfinite(D)] = 0
    A = csgraph.laplacian(A, use_out_degree=True, normed=False)
    return A,D
def complete_Graph_features(inp):
    coordinates = inp.COORDINATES
    outfall_attrs = inp.OUTFALLS.frame.iloc[:,:1]
    outfall_attrs['depth_max']=0
    node_elevaltion_attrs=pd.concat([inp.JUNCTIONS.frame.iloc[:,:2],inp.STORAGE.frame.iloc[:,:2],outfall_attrs],axis=0)
    node_list = list(coordinates.keys())

    links = {(val.from_node, val.to_node, 0): {'in_offset': val.offset_upstream,
                                                  'length': val.length,
                                                  'name_conduits': val.name,
                                                  'out_offset': val.offset_downstream,
                                                  'roughness': val.roughness,
                                                  'is_pump': 0.0,
                                                  'depth_on': np.nan,
                                                  'depth_off': np.nan,
                                                  } for key, val in
                inp.CONDUITS.items()}
    links.update({(val.from_node, val.to_node, int(val.name.split('.')[-1])): {'in_offset': np.nan,
                                                                          'length': np.nan,
                                                                          'name_conduits': val.name,
                                                                          'out_offset': np.nan,
                                                                          'roughness': np.nan,
                                                                          'is_pump': 1,
                                                                          'depth_on': val.depth_on,
                                                                          'depth_off': val.depth_off,
                                                                          } for key, val in
             inp.PUMPS.items()})

    x_sections = {key: {'conduit_shape': 0.0, 'height': val.height} for key, val in
                  inp.XSECTIONS.items()} | {key: {'conduit_shape': np.nan, 'height': np.nan} for key, val
                                                         in inp.PUMPS.items()}
    for (k, v), (a,b) in zip(links.items(), x_sections.items()):
        v.update(b)
    return node_list,node_elevaltion_attrs,coordinates,links
def true_idx(representative_nodes):
    monitored_ids = [modify_node_list[i] for i in representative_nodes]
    all_node_list = coordinates.frame.index.tolist()
    monitored_idx_map = [all_node_list.index(i) for i in monitored_ids]
    return monitored_idx_map
def select_sensors_algorithm(L, num_sensors,Psi):
    selected = []
    candidate_indices = list(range(len(Psi)))
    epsilon = 1e-6
    for _ in range(num_sensors):
        scores = []
        for i in candidate_indices:
            redundancy = 0
            if selected:
                redundancy = np.mean([compute_cgir(Psi[i], Psi[j]) for j in selected])
            score = np.mean(Psi)*np.linalg.norm(Psi[i], ord=1, axis=0) / (epsilon + redundancy)
            scores.append(score)
        best_idx = candidate_indices[np.argmax(scores)]
        selected.append(best_idx)
        candidate_indices.remove(best_idx)
    return selected
if __name__ == '__main__':
    # === Step 1: 读取 INP 文件并解析节点与管道属性 ===
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    model_pt = current_dir/'garage'
    network_name='caoyang'
    inp = read_inp_file(current_dir/'data'/'SWMM_data'/f'{network_name}'/'networks'/f"{network_name}.inp")
    test_data_path = current_dir/'data'/'Results_data'

    node_list,node_elevaltion_attrs,coordinates,links = complete_Graph_features(inp)
    G = get_Graph(node_list,coordinates,links)
    G_idx = get_Graph_Idx(node_list, coordinates, links)
    source_node = 'J04030201020701151596'
    need_delet_nodes = list(nx.descendants(G, source_node))

    modify_node_list= [n for n in node_list if n not in need_delet_nodes]

    # ===== STEP 3: 构建 PyG 图数据结构 =====
    edges,edge_attrs=edge_attr_extraction(links,modify_node_list,need_delet_nodes,node_elevaltion_attrs)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    L, adj = get_length_laplacian()
    Psi = calculate_Psi(L)

    sensor_counts=[]
    coverages = []
    sensor_sets=[]
    for num in range(1,21):
        graph_sensors_sets = select_sensors_algorithm(L,num,Psi)
        map_sensors = true_idx(graph_sensors_sets)
        sensor_covered_layout(G_idx, map_sensors)
        sensor_counts.append(len(map_sensors))
        sensor_sets.append(map_sensors)
    sensor_dict ={'sensor_counts':sensor_counts,'coverages':coverages,'sensor_sets':sensor_sets}
    save_path = current_dir/'data'/'Results_data'/'4.1PD_optimization_layout_results'/'Graphy'
    import pickle
    save_path.mkdir(parents=True,exist_ok=True)
    with open(save_path/"10layer_layouts_start25_Greedy.pkl", "wb") as f:
        pickle.dump(sensor_dict, f)

    map_sensors = true_idx(graph_sensors_sets)

    sensors_other = true_idx(random.sample(
        list(set([i for i in range(len(modify_node_list)) if i not in map_sensors]) - set(map_sensors)), 10))

    print(graph_sensors_sets)
    print(sensors_other)
    # idx_map_complete = true_idx(baseline_sensors_sets)
    # # Compute Chebyshev coefficients and operator
    # r = 20
    # alpha = compute_chebyshev_coeff(L, r)
    # Psi = chebyshev_polynomial_operator(L, alpha)
    #
    # real_values = run_model_in_testing_event(test_data_path)
    # f_C = real_values[0][1000,idx_map_complete,:]
    # unselected_nodes = true_idx([i for i in range(len(modify_node_list)) if i not in baseline_sensors_sets])
    # f_R_true = real_values[0][1000,unselected_nodes,:]
    # # Reconstruct pressure at unmonitored nodes
    # f_R = np.array(reconstruct_pressure_signal(Psi, f_C, baseline_sensors_sets))
    # error = np.linalg.norm(np.array(f_R) - f_R_true, ord=2) ** 2
    # print(f"monitored_nodes: {idx_map_complete}")



    # Step 2: 特征分解 (L = U Λ U^T)

    #     plt.plot(range(1, L.shape[0]+1), psi_i, marker='o', label=f'Center Node v{i+1}')
    #
    # plt.title('Graph Localization Operator Response')
    # plt.xlabel('Node Index')
    # plt.ylabel('Localization Value ψ')
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(range(1, 36))
    # plt.show()
