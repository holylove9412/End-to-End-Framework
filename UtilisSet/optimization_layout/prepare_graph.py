import numpy as np
import networkx as nx
from swmm_api.input_file import read_inp_file
from swmm_api.input_file.section_labels import JUNCTIONS,CONDUITS, CONDUITS
from pathlib import Path
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
font1 = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
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
def get_Graph(node_ids,coordinates,links):
    G = nx.MultiDiGraph()
    for node_id in node_ids:
        G.add_node(node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']))
    for idx, row in links.items():
        from_node, to_node = idx[0], idx[1]
        G.add_edge(from_node, to_node)
    return G
def get_G_without_pump(manholes,pipes,coordinates):
    G = nx.Graph()
    for i, node_id in enumerate(manholes):
        G.add_node(i, label=node_id, pos=(coordinates[node_id]['x'], coordinates[node_id]['y']))
    for from_idx, to_idx in pipes:
        G.add_edge(from_idx, to_idx)
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
        edge_attrs.append([diameter, length, slope,capacity_score])
    return edges,edge_attrs
def get_upstream_node_sets(G):
    upstream_dict = {}
    upstream_length = {}
    for target_node in G.nodes:
        upstream_set = nx.ancestors(G, target_node)
        upstream_set.add(target_node)
        upstream_dict[target_node] = upstream_set
        upstream_length[target_node] = len(upstream_set)
    return upstream_dict,upstream_length
def generate_conn_dict_from_graph(G):
    conn_dict = {}
    for i in G.nodes:
        for j in G.nodes:
            if nx.has_path(G, i, j):
                conn_dict[(i, j)] = 1
    for node in G.nodes():
        conn_dict[(node, node)] = 1
    return conn_dict
def get_graph_withou_pump(inp_file_path):
    print(f"Loading .inp file: {inp_file_path}")
    # G = build_graph_from_inp(inp_file_path)
    inp_file = read_inp_file(inp_file_path)
    node_list,node_elevaltion_attrs,coordinates,links = complete_Graph_features(inp_file)
    G_with_pump = get_Graph(node_list,coordinates,links)

    source_node = 'J04030201020701151596'
    need_delet_nodes = list(nx.descendants(G_with_pump, source_node))

    modify_node_list= [n for n in node_list if n not in need_delet_nodes]

    edges,edge_attrs=edge_attr_extraction(links,modify_node_list,need_delet_nodes,node_elevaltion_attrs)
    # edges, edge_attrs = edge_attr_extraction(links, node_list, [], node_elevaltion_attrs)
    G = get_G_without_pump(modify_node_list,edges,coordinates)
    return G,modify_node_list,node_list
def sensor_covered_layout(G,sensor_node,covered_nodes,fig_path=None):

    pos = nx.get_node_attributes(G, 'pos')

    sensor_color = ['lime' for _ in range(len(sensor_node))]
    covered_color = ['#2c4ca0' for _ in range(len(covered_nodes))]


    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_size=100, node_color='#d2e2ef', font_size=8, font_color='black')

    nx.draw_networkx_nodes(G, pos, nodelist=covered_nodes, edgecolors='black', node_size=100,
                           node_color=covered_color)

    nx.draw_networkx_nodes(G, pos, nodelist=sensor_node, node_shape='*', edgecolors='black',node_size=700, node_color=sensor_color)
    # plt.savefig(fig_path,dpi=800)
    plt.show()
if __name__ == "__main__":

    random.seed(42)
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    model_pt = current_dir/'garage'
    network_name='study_area'
    inp_path = read_inp_file(current_dir/'data'/'SWMM_data'/f'{network_name}'/'networks'/f"{network_name}.inp")

    get_graph_withou_pump(inp_path)
