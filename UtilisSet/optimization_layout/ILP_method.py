import pulp
from prepare_graph import *
import random
def optimal_sensor_placement(G, k=11):

    sp_lengths = dict(nx.all_pairs_shortest_path_length(G))

    neighbors_k = {
        node: {n for n, d in dist.items() if d <= k}
        for node, dist in sp_lengths.items()
    }

    model = pulp.LpProblem("Optimal_Sensor_Placement", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", G.nodes, cat="Binary")
    model += pulp.lpSum(x[i] for i in G.nodes)

    for i in G.nodes:
        model += pulp.lpSum(x[j] for j in neighbors_k[i]) >= 1

    solver = pulp.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    sensors = [i for i in G.nodes if pulp.value(x[i]) == 1]

    all_covered = set()
    for n in sensors:
        best_coverage = neighbors_k[n]
        all_covered |= best_coverage

    return sensors, all_covered,neighbors_k

    # return sensors
def true_idx(representative_nodes):
    monitored_ids = [modify_node_list[i] for i in representative_nodes]

    monitored_idx_map = [node_list.index(i) for i in monitored_ids]
    return monitored_idx_map
if __name__ == "__main__":
    random.seed(42)
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    network_name='study_area'
    inp_path = current_dir/'data'/'SWMM_data'/f'{network_name}'/'networks'/f"{network_name}.inp"

    fig_path = current_dir/'data'/'Results_data'/'results_pic'/'domination_layout'
    fig_path.mkdir(parents=True, exist_ok=True)

    G_without_pump,modify_node_list,node_list = get_graph_withou_pump(inp_path)
    sensors,coverage,cover_sets = optimal_sensor_placement(G_without_pump,k=10)

    sensor_covered_layout(G_without_pump, sensors, coverage)
    sensors_other = random.sample(
        list(set([i for i in range(len(modify_node_list)) if i not in sensors]) - set(sensors)), 10)

    revise_sensors = true_idx(sensors)
    temple_sensors = true_idx(sensors_other)
    print(f"Selected sensor nodes: {revise_sensors}")
    print(f"sensing domains percent: {temple_sensors}")
