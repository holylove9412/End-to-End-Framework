import numpy as np
import networkx as nx
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pickle
from prepare_graph import *
from greedy_ga import *
from new_ga import *
font = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
font1 = fm.FontProperties(fname=r"C:\Windows\Fonts\times.ttf", size=20)
class SensorPlacementGA:
    def __init__(self, graph, population_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.05, sensor_upbound=None):
        self.graph = graph
        self.node_num = len(graph.nodes)
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.sensor_upbound = sensor_upbound if sensor_upbound else self.node_num
        self.K = 11

        self.khop_neighbors = {n: set(nx.single_source_shortest_path_length(graph, n, cutoff=self.K).keys()) for n in graph.nodes}

    def _init_population(self):
        return np.random.randint(0, 2, (self.population_size, self.node_num))

    def _fitness(self, chromosome):
        selected = np.where(chromosome == 1)[0]
        covered_nodes = set()
        for s in selected:
            covered_nodes |= self.khop_neighbors[s]

        coverage_ratio = len(covered_nodes) / self.node_num
        sensor_ratio = len(selected) / self.node_num

        if coverage_ratio == 1:
            return sensor_ratio
        else:
            return sensor_ratio + 10 * (1 - coverage_ratio)

    def _selection(self, population, fitness):
        idx1, idx2 = np.random.randint(0, self.population_size, 2)
        return population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, self.node_num - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutation(self, chromosome):
        for i in range(self.node_num):
            if np.random.rand() < self.mutation_prob:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def run(self):
        population = self._init_population()
        best_fitness_list = []

        for gen in range(self.generations):
            fitness = np.array([self._fitness(ind) for ind in population])
            new_population = []

            best_idx = np.argmin(fitness)
            best_individual = population[best_idx].copy()
            best_fitness = fitness[best_idx]
            best_fitness_list.append(best_fitness)

            while len(new_population) < self.population_size:
                parent1 = self._selection(population, fitness)
                parent2 = self._selection(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population)
            population[0] = best_individual

            print(f"Generation {gen+1}, Best Fitness: {best_fitness:.4f}")

        final_fitness = np.array([self._fitness(ind) for ind in population])
        best_idx = np.argmin(final_fitness)
        best_solution = population[best_idx]
        selected_sensors = np.where(best_solution == 1)[0]
        all_covered = set()
        for n in selected_sensors:
            best_coverage = self.khop_neighbors[n]
            all_covered |= best_coverage

        return selected_sensors, all_covered
def sensor_covered_layout(G,sensor_node,covered_nodes,sensing_percent):

    pos = nx.get_node_attributes(G, 'pos')

    sensor_color = ['lime' for _ in range(len(sensor_node))]
    covered_color = ['#2c4ca0' for _ in range(len(covered_nodes))]


    plt.figure(figsize=(11, 11))
    nx.draw(G, pos, node_size=100, node_color='#d2e2ef', font_size=8, font_color='black')

    nx.draw_networkx_nodes(G, pos, nodelist=covered_nodes, edgecolors='black', node_size=100,
                           node_color=covered_color)

    nx.draw_networkx_nodes(G, pos, nodelist=sensor_node, node_shape='*', edgecolors='black',node_size=700, node_color=sensor_color)
    plt.savefig(fig_path/f'sensing_{sensing_percent}.png',dpi=800)
    plt.show()

def pre_read():
    G,modify_node_list,node_list = get_graph_withou_pump(inp_path)
    upstream_sets,upstream_arrs = get_upstream_node_sets(G)
    node_nums = len(G.nodes())
    conn_dicts = generate_conn_dict_from_graph(G)

    return upstream_arrs, upstream_sets,G,conn_dicts,node_nums,[modify_node_list,node_list]
def true_idx(representative_nodes,node_idx_map):
    modify_node_list, node_list = node_idx_map[0],node_idx_map[1]
    monitored_ids = [modify_node_list[i] for i in representative_nodes]

    monitored_idx_map = [node_list.index(i) for i in monitored_ids]
    return monitored_idx_map
if __name__ == '__main__':
    random.seed(42)
    current_file = Path(__file__)
    current_dir = current_file.parent.parent.parent
    model_pt = current_dir/'garage'
    systerm_name='study_area'
    inp_path =current_dir/'data'/'SWMM_data'/f'{systerm_name}'/'networks'/f"{systerm_name}.inp"
    fig_path = current_dir/'data'/'Results_data'/'results_pic'/'domination_layout'
    fig_path.mkdir(parents=True, exist_ok=True)

    upstream_arr, upstream_set, relabeled_G, conn_dict, node_num,node_idx = pre_read()

    sensor_counts, coverages, sensor_sets = run_multi_upperbound(relabeled_G, start=25, step=1)
    save_data = {'sensor_counts':sensor_counts,'coverages':coverages,'sensor_sets':sensor_sets}
    save_path = current_dir/'data'/'Results_data'/'PD_optimization_layout_results'/'GA'
    save_path.mkdir(parents=True,exist_ok=True)
    with open(save_path/"10layer_layouts_start25.pkl", "wb") as f:
        pickle.dump(save_data, f)

    plt.figure(figsize=(8, 6))
    plt.plot(sensor_counts, coverages, marker="o")
    plt.xlabel("Sensor Count")
    plt.ylabel("Coverage Ratio")
    plt.title("Coverage vs Sensor Count")
    plt.grid()
    plt.show()
