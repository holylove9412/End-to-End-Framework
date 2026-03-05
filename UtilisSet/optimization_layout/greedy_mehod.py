from prepare_graph import *
import pickle
font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
font1 = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)

def sensor_covered_layout(G,sensor_node,covered_nodes,sensing_percent):

    pos = nx.get_node_attributes(G, 'pos')

    sensor_color = ['lime' for _ in range(len(sensor_node))]
    covered_color = ['#2c4ca0' for _ in range(len(covered_nodes))]


    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_size=100, node_color='#d2e2ef', font_size=8, font_color='black')

    nx.draw_networkx_nodes(G, pos, nodelist=covered_nodes, edgecolors='black', node_size=100,
                           node_color=covered_color)

    nx.draw_networkx_nodes(G, pos, nodelist=sensor_node, node_shape='*', edgecolors='black',node_size=700, node_color=sensor_color)
    plt.tight_layout()
    plt.savefig(fig_path/f'sensing_{sensing_percent}.png',dpi=800)
    plt.show()


def fast_prune_redundant_coverage_sets(coverage_sets: dict):

    sorted_nodes = sorted(coverage_sets.items(), key=lambda x: -len(x[1]))

    representative_sets = []
    non_redundant_nodes = []

    for node, S in sorted_nodes:
        is_redundant = False
        for _, R in representative_sets:
            if S <= R:
                is_redundant = True
                break
        if not is_redundant:
            representative_sets.append((node, S))
            non_redundant_nodes.append(node)
    tt=coverage_sets.keys()-set(non_redundant_nodes)
    return non_redundant_nodes,representative_sets

def greedy_sensor_selection_by_max_coverage(coverage_sets_inita: dict,total_coverage_sets,sensing_domains):
    coverage_sets = {k: v.copy() for k, v in coverage_sets_inita.items()}  # 深拷贝，避免修改原数据
    selected_sensors = []
    all_covered = set()
    sensors_counts = []
    sensors_coverages = []
    selected_sensors_sets=[]

    while coverage_sets:
        best_node = max(coverage_sets, key=lambda n: len(coverage_sets[n]))
        best_coverage = coverage_sets[best_node]
        selected_sensors.append(best_node)
        all_covered |= best_coverage

        sensors_counts.append(len(selected_sensors))
        sensors_coverages.append(len(all_covered)/len(modify_node_list))
        selected_sensors_sets.append(selected_sensors.copy())

        coverage_sets = {
            n: s for n, s in coverage_sets.items() if n not in all_covered
        }
        for n in coverage_sets:
            coverage_sets[n] = coverage_sets[n]-best_coverage

        if len(coverage_sets)==0 and len(all_covered)/len(modify_node_list)<1:
            branch_end_nodes = list(set(range(len(modify_node_list)))-set(all_covered))
            branch_end_sets = {key : total_coverage_sets[key] for key in branch_end_nodes}
            coverage_sets.update(branch_end_sets)

    sensors_dict = {'sensor_counts':sensors_counts,'coverages':sensors_coverages,'sensor_sets':selected_sensors_sets}

    return selected_sensors,all_covered,sensors_dict
def compute_sensor_layout_min_overlap(G, depth=10,sensing_percent=1):
    coverage_sets = {
        node: set(nx.single_source_shortest_path_length(G, node, cutoff=depth).keys())
        for node in G.nodes
    }
    candidate_nodes,candidate_sets = fast_prune_redundant_coverage_sets(coverage_sets)

    selected_sensors, coverage_nodes,sensors_dict = greedy_sensor_selection_by_max_coverage(dict(candidate_sets), coverage_sets,
                                                                               sensing_percent)

    sensor_covered_layout(G,selected_sensors,coverage_nodes,sensing_percent)

    return selected_sensors, coverage_nodes,sensors_dict
def true_idx(representative_nodes,node_idx_map):
    modify_node_list, node_list = node_idx_map[0],node_idx_map[1]
    monitored_ids = [modify_node_list[i] for i in representative_nodes]

    monitored_idx_map = [node_list.index(i) for i in monitored_ids]
    return monitored_idx_map

if __name__ == "__main__":
    random.seed(42)
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    model_pt = current_dir/'garage'
    network_name='study_area'
    inp_file = read_inp_file(current_dir/'data'/'SWMM_data'/f'{network_name}'/'networks'/f"{network_name}.inp")
    inp_path = current_dir / 'data' / 'SWMM_data' / f'{network_name}' / 'networks' / f"{network_name}.inp"

    fig_path = current_dir/'data'/'Results_data'/'results_pic'/'domination_layout'
    fig_path.mkdir(parents=True, exist_ok=True)

    # node_list,node_elevaltion_attrs,coordinates,links = complete_Graph_features(inp_file)

    G_without_pump,modify_node_list,node_list = get_graph_withou_pump(inp_path)
    # for percent in [0.5,0.6,0.7,0.8,0.9,0.95,0.98,1]:
    sensors, coverage,sensors_dict = compute_sensor_layout_min_overlap(G_without_pump, depth=10,sensing_percent=1)
    sensors = [134, 216, 105, 24, 90, 224, 79, 272, 295, 61, 152, 343, 333, 375, 314, 8, 386, 165, 293]

    save_path = current_dir/'data'/'Results_data'/'4.1PD_optimization_layout_results'/'Greedy'

    save_path.mkdir(parents=True,exist_ok=True)
    with open(save_path/"10layer_layouts_start25_Greedy.pkl", "wb") as f:
        pickle.dump(sensors_dict, f)

    map_sensors = true_idx(sensors,[modify_node_list,node_list])

    sensors_other = true_idx(random.sample(
        list(set([i for i in range(len(modify_node_list)) if i not in map_sensors]) - set(map_sensors)), 10),[modify_node_list,node_list])

    print(f"Selected sensor nodes: {map_sensors}")
    print(f"Selected sensor other nodes: {sensors_other}")
    print(f"Total sensors: {len(sensors)}")

    plt.figure(figsize=(8, 6))
    plt.plot(sensors_dict['sensor_counts'], sensors_dict['coverages'], marker="o")
    plt.xlabel("Sensor Count")
    plt.ylabel("Coverage Ratio")
    plt.title("Coverage vs Sensor Count")
    plt.grid()
    plt.show()


