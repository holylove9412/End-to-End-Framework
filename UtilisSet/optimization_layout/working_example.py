import networkx as nx
from swmm_api.input_file import read_inp_file
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import random

font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
font1 = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
def complete_Graph_features(inp):
    coordinates = inp.COORDINATES
    outfall_attrs = inp.OUTFALLS.frame.iloc[:,:1]
    outfall_attrs['depth_max']=0
    area={}
    for k,s in inp.SUBCATCHMENTS.items():
        area[s['outlet']]=round(s['area'],2)
    area = pd.DataFrame(area,index=['area']).T
    node_elevaltion_attrs=pd.concat([inp.JUNCTIONS.frame.iloc[:,:2],inp.STORAGE.frame.iloc[:,:2],outfall_attrs],axis=0)

    node_elevaltion_attrs=pd.concat([node_elevaltion_attrs,area],axis=1)
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
def get_Graph(node_ids,coordinates,links):
    G = nx.MultiDiGraph()
    for node_id in node_ids:
        G.add_node(node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']))
    for idx, row in links.items():
        from_node, to_node = idx[0], idx[1]
        G.add_edge(from_node, to_node)
    return G
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
        diameter = row['height']
        length = row['length']
        slope_conduit = (row['in_offset']-row['out_offset'])/length
        z1 = node_elevaltion_attrs.loc[from_node]['elevation']
        z2 = node_elevaltion_attrs.loc[to_node]['elevation']
        # slope = max((z1 - z2) / length, 0.0001)
        slope = (z1 - z2) / length
        capacity_score = (diameter ** 2) / length * slope
        edges.append((from_idx, to_idx))
        # edge_attrs.append([diameter, length, slope,capacity_score])
        edge_attrs.append([diameter, length,row['in_offset'],row['out_offset'], capacity_score])
    return edges,edge_attrs

def get_G_without_pump(manholes,pipes,pipes_attr):
    G = nx.DiGraph()
    for i, node_id in enumerate(manholes):
        G.add_node(i, label=node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']))
    for (from_idx, to_idx),d in zip(pipes,pipes_attr ):
        G.add_edge(from_idx, to_idx,diameter=d[0],length=d[1])
    return G

def edge_us_with_depth(G, start_node, max_depth,key_word):
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
            if key_word=='diameter':
                diameter = G.edges[pred, node][key_word]
            if key_word=='invert':
                diameter = G.edges[pred, node][key_word][0]
            if key_word=='length':
                diameter = G.edges[pred, node][key_word]
            edge_us.append((pred, node, diameter, depth))
            if pred not in visited:
                stack.append((pred, depth + 1))
    return edge_us

def edge_ds_with_depth(G, start_node, max_depth,key_word):
    visited = set()
    edge_ds = []
    # edge_weights = []
    stack = [(start_node, 0)]  # (current_node, current_depth)
    while stack:
        node, depth = stack.pop()
        if depth > max_depth:
            continue
        visited.add(node)
        for pred in G.successors(node):
            if key_word=='diameter':
                diameter = G.edges[node, pred][key_word]
            if key_word=='invert':
                diameter = G.edges[node, pred][key_word][-1]
            if key_word == 'length':
                diameter = G.edges[node, pred][key_word]
            edge_ds.append((node, pred, diameter, depth))
            if pred not in visited:
                stack.append((pred, depth + 1))
    return edge_ds
def compute_edge_weight(diameter, depth, diameter_mean,diameter_std, depth_std, alpha=0.8):
    # diameter_weight = np.exp(-(diameter / (diameter_std + 1e-5)) ** 2)

    diameter_weight = np.exp(-((diameter-diameter_mean) / (diameter_std + 1e-5)) ** 2)

    depth_weight = np.exp(-(depth / (depth_std + 1e-5)) ** 2)
    combined_weight = alpha * diameter_weight + (1 - alpha) * depth_weight
    return combined_weight
def true_idx(representative_nodes):
    monitored_ids = [modify_node_list[i] for i in representative_nodes]

    all_node_list = list(coordinates.keys())
    monitored_idx_map = [all_node_list.index(i) for i in monitored_ids]
    return monitored_idx_map

if __name__ == "__main__":
    random.seed(42)
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    model_pt = current_dir/'garage'
    network_name='study_area'
    inp_file = read_inp_file(current_dir/'data'/'SWMM_data'/f'{network_name}'/'networks'/f"{network_name}.inp")

    fig_path = current_dir/'data'/'Results_data'/'results_pic'/'domination_layout'
    fig_path.mkdir(parents=True, exist_ok=True)

    node_list,node_elevaltion_attrs,coordinates,links = complete_Graph_features(inp_file)

    ##get node properties
    row_name = list(node_elevaltion_attrs.index)
    rename_row_name = [node_list.index(i) for i in row_name]
    node_elevaltion_attrs.index=rename_row_name
    df = node_elevaltion_attrs.sort_index()
    df.to_csv(current_dir / 'data' / 'Results_data' / 'node_attrs.csv',index=True)

    G_with_pump = get_Graph(node_list,coordinates,links)
    source_node = 'J04030201021200471815'
    need_delet_nodes = list(nx.descendants(G_with_pump, source_node))

    modify_node_list= [n for n in node_list if n not in need_delet_nodes]

    edges,edge_attrs=edge_attr_extraction(links,node_list,[],node_elevaltion_attrs)
    G_without_pump = get_G_without_pump(node_list, edges,edge_attrs)

    matrix_L =np.zeros((len(node_list),len((node_list))))
    depth =1
    key_word = 'diameter' ##get matrixL and matrix D
    for n in G_without_pump.nodes():
        edges_us,edges_ds = edge_us_with_depth(G_without_pump,n,depth,key_word),edge_ds_with_depth(G_without_pump,n,depth,key_word)
        connect_diameter = [edge[2] for edge in edges_us + edges_ds]
        d_std = np.std(connect_diameter)
        d_mean = np.mean(connect_diameter)
        for i,us in enumerate(edges_us):
            edges_weight = compute_edge_weight(us[2],us[3]+1,d_mean,d_std,depth+1,alpha=0.5)
            matrix_L[n,us[0]] = round(edges_weight,2)
        for i,ds in enumerate(edges_ds):
            edges_weight = compute_edge_weight(ds[2], ds[3] + 1, d_mean, d_std, depth+1, alpha=0.5)
            matrix_L[n, ds[1]] = round(edges_weight,2)

    pd.DataFrame(matrix_L).to_csv(current_dir / 'data' / 'Results_data' / 'matrix_D.csv')

    table = np.concatenate((np.array([[i[0],i[1]]for i in edges]),np.array(edge_attrs)),axis=1)
    conduit_df = pd.DataFrame(table,columns=['from_node','to_node','diameter','length','us_invert','ds_invert','capacity_score'])
    conduit_df.to_excel(current_dir/'data'/'Results_data'/'conduit_attrs.xls')

    matrxi_G = np.zeros((len(node_list),len(edges)))

    for i,((u,v),j) in enumerate(zip(edges,edge_attrs)):
        matrxi_G[u,i] = round(j[2],2)
        matrxi_G[v, i] = round(j[3],2)
    pd.DataFrame(matrxi_G).to_csv(current_dir / 'data' / 'Results_data' / 'matrix_G.csv')

    matrxi_V = np.zeros((len(node_list),len(edges)))

    for i,((u,v),j) in enumerate(zip(edges,edge_attrs)):
        matrxi_V[u,i] = 1
        matrxi_V[v, i] = -1
    pd.DataFrame(matrxi_V).to_csv(current_dir / 'data' / 'Results_data' / 'matrix_V.csv')

    matrix_A = nx.to_numpy_array(G_with_pump)
    pd.DataFrame(matrix_A).to_csv(current_dir / 'data' / 'Results_data' / 'matrix_A.csv')
