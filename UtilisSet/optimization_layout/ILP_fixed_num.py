import pulp
import pickle
from prepare_graph import *
import matplotlib.pyplot as plt
def coverage_vs_sensors(G, k=10, max_sensors=None):
    nodes = list(G.nodes())
    n = len(nodes)
    if max_sensors is None:
        max_sensors = n

    neighbors_k = {
        i: set(nx.single_source_shortest_path_length(G, i, cutoff=k).keys())
        for i in nodes
    }

    results = []

    for m in range(1, max_sensors + 1):
        model = pulp.LpProblem("Coverage_Maximization", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("x", nodes, 0, 1, cat="Binary")
        y = pulp.LpVariable.dicts("y", nodes, 0, 1, cat="Binary")

        model += pulp.lpSum(y[i] for i in nodes)

        for i in nodes:
            model += pulp.lpSum(x[j] for j in neighbors_k[i]) >= y[i]

        model += pulp.lpSum(x[i] for i in nodes) <= m

        # 求解
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        covered = sum(pulp.value(y[i]) for i in nodes)/len(modify_node_list)
        sensors = [i for i in G.nodes if pulp.value(x[i]) == 1]
        results.append((m, covered,sensors))

    return results
def true_idx(representative_nodes):
    monitored_ids = [modify_node_list[i] for i in representative_nodes]

    monitored_idx_map = [node_list.index(i) for i in monitored_ids]
    return monitored_idx_map
if __name__ == "__main__":
    random.seed(42)
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    model_pt = current_dir/'garage'
    network_name='study_area'
    inp_path = current_dir/'data'/'SWMM_data'/f'{network_name}'/'networks'/f"{network_name}.inp"

    fig_path = current_dir/'data'/'Results_data'/'results_pic'/'domination_layout'
    fig_path.mkdir(parents=True, exist_ok=True)

    G_without_pump,modify_node_list,node_list = get_graph_withou_pump(inp_path)
    curve = coverage_vs_sensors(G_without_pump,k=10,max_sensors=25)
    sensors = true_idx(curve[14][2])
    sensors_other = true_idx(random.sample(
        list(set([i for i in range(len(modify_node_list)) if i not in sensors]) - set(sensors)), 10))

    sensors_counts = [i[0] for i in curve]
    sensors_coverages = [i[1] for i in curve]
    selected_sensors = [i[2] for i in curve]
    sensors_dict = {'sensor_counts': sensors_counts, 'coverages': sensors_coverages, 'sensor_sets': selected_sensors}

    save_path = current_dir/'data'/'Results_data'/'PD_optimization_layout_results'/'ILP'
    save_path.mkdir(parents=True,exist_ok=True)
    with open(save_path/"10layer_layouts_start25.pkl", "wb") as f:
        pickle.dump(sensors_dict, f)

    m_vals, cov_vals,sensors = zip(*curve)
    plt.plot(m_vals, cov_vals, marker="o")
    plt.xlabel("Sensor Count")
    plt.ylabel("Coverage Ratio")
    plt.title("Coverage vs Sensor Count")
    plt.grid()
    plt.show()

