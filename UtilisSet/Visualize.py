import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from swmm_api import read_inp_file
import matplotlib.font_manager as fm

import matplotlib.cm as cm

font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
font1 = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)

def visualize_features(G, features, title="Node Features"):
    pos = nx.get_node_attributes(G, 'pos')
    # pos = nx.spring_layout(G)
    all_zero_mask = (features[:,:1] != 0).all(axis=-1)
    node_label = all_zero_mask.astype(int)

    fig, ax = plt.subplots(figsize=(10, 8))

    norm = colors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    nx.draw(G, pos, with_labels=True, node_size=60, font_size=8,node_color=node_label, cmap='viridis', ax=ax)
    plt.title(title)

    fig.colorbar(sm, ax=ax, label="Feature Value")
    plt.show()
def nx_plot(label_nodes,nse,swmm_inp,save_dir=None):

    nse[nse<-0.1]=-0.1
    inp = read_inp_file(swmm_inp)
    coordinates = inp.COORDINATES
    nodes = inp.JUNCTIONS
    conduits = inp.CONDUITS
    outfalls = inp.OUTFALLS
    storages = inp.STORAGE
    X = nx.DiGraph()
    for node in nodes:
        X.add_node(node, pos=(coordinates[node]['x'], coordinates[node]['y']))
    for outfall in outfalls:
        X.add_node(outfall, pos=(coordinates[outfall]['x'], coordinates[outfall]['y']))
    for storage in storages:
        X.add_node(storage, pos=(coordinates[storage]['x'], coordinates[storage]['y']))
    for cunduit in conduits:
        inlet = conduits[cunduit]['from_node']
        outlet = conduits[cunduit]['to_node']
        if cunduit in ['JTN-03(tb2).1','JTN-03(tb).2']:
            pass
        else:
            X.add_edge(inlet, outlet)

    pos = nx.get_node_attributes(X, 'pos')
    node_ids = list(coordinates.keys())

    unmask_node = {f'{node_ids[i]}' for i in label_nodes}
    unmask_color = ['lime' for _ in range(len(unmask_node))]
    parula_map = 'RdYlBu'

    plt.figure(figsize=(11, 11))
    nx.draw(X, pos, with_labels=False, node_size=100, node_color=nse, cmap=parula_map,font_size=8, font_color='black')

    nx.draw_networkx_nodes(X, pos, nodelist=unmask_node, node_shape='*', edgecolors='black',node_size=700, node_color=unmask_color)

    sm = plt.cm.ScalarMappable(cmap=parula_map,norm=plt.Normalize(vmin=min(nse),vmax=1))
    sm.set_array([])
    cbar_ax = plt.gcf().add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(sm,cax=cbar_ax)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(font)
    tick_values = np.arange(0,1.1,0.1)
    tick_values = np.insert(tick_values,0,-0.1)
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val:.1f}' for val in tick_values])

    plt.title('NSE',fontproperties=font)

    if save_dir:
        plt.savefig(save_dir,dpi=300)
        plt.clf()
    else:
        plt.show()