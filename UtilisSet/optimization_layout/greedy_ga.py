import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt

class HybridSensorPlacement:
    def __init__(self, graph, K=10, population_size=50, generations=150,
                 crossover_prob=0.8, mutation_prob=0.05):
        self.graph = graph
        self.node_num = len(graph.nodes)
        self.K = K
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob

        self.khop_neighbors = {
            n: set(nx.single_source_shortest_path_length(graph, n, cutoff=K).keys())
            for n in graph.nodes
        }

    def greedy_placement(self, target_coverage=1.0):
        uncovered = set(self.graph.nodes)
        sensors = []
        while len(uncovered) / self.node_num > (1 - target_coverage):
            best_node = max(uncovered, key=lambda n: len(self.khop_neighbors[n] & uncovered))
            sensors.append(best_node)
            uncovered -= self.khop_neighbors[best_node]
        return sensors

    def _fitness(self, chromosome, target_coverage):
        selected = np.where(chromosome == 1)[0]
        covered = set()
        for s in selected:
            covered |= self.khop_neighbors[s]
        coverage_ratio = len(covered) / self.node_num

        if coverage_ratio < target_coverage:
            return 1e6 + (target_coverage - coverage_ratio) * 1000
        return len(selected)

    def _init_population(self, greedy_solution):
        population = []
        for _ in range(self.population_size):
            chrom = np.zeros(self.node_num, dtype=int)
            chrom[greedy_solution] = 1
            flip_num = max(1, int(0.1 * len(greedy_solution)))
            flip_nodes = random.sample(range(self.node_num), flip_num)
            chrom[flip_nodes] = 1 - chrom[flip_nodes]
            population.append(chrom)
        return np.array(population)

    def _selection(self, population, fitness):
        idx1, idx2 = np.random.randint(0, self.population_size, 2)
        return population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

    def _crossover(self, p1, p2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, self.node_num - 1)
            return np.concatenate((p1[:point], p2[point:])), np.concatenate((p2[:point], p1[point:]))
        return p1.copy(), p2.copy()

    def _mutation(self, chrom):
        for i in range(self.node_num):
            if np.random.rand() < self.mutation_prob:
                chrom[i] = 1 - chrom[i]
        return chrom

    def ga_refinement(self, greedy_solution, target_coverage):
        population = self._init_population(greedy_solution)
        best_solution = greedy_solution
        best_fitness = len(greedy_solution)

        for _ in range(self.generations):
            fitness = np.array([self._fitness(ind, target_coverage) for ind in population])
            new_population = []

            elite_num = max(1, int(0.05 * self.population_size))
            elite_idx = np.argsort(fitness)[:elite_num]
            elites = population[elite_idx]
            new_population.extend(elites)

            while len(new_population) < self.population_size:
                p1 = self._selection(population, fitness)
                p2 = self._selection(population, fitness)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutation(c1)
                c2 = self._mutation(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)

            population = np.array(new_population)

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = np.where(population[best_idx] == 1)[0]
                best_fitness = fitness[best_idx]

        return best_solution

    def run_multi_threshold(self, thresholds=[1.0, 0.95, 0.9, 0.85, 0.8]):
        sensor_counts, coverages, sensor_sets = [], [], []

        for target in thresholds:
            greedy_solution = self.greedy_placement(target)

            refined_solution = self.ga_refinement(greedy_solution, target)

            covered = set()
            for s in refined_solution:
                covered |= self.khop_neighbors[s]
            coverage_ratio = len(covered) / self.node_num

            sensor_counts.append(len(refined_solution))
            coverages.append(coverage_ratio)
            sensor_sets.append(refined_solution)

            # 输出结果（打印传感器编号）
            print("=" * 60)
            print(f"目标覆盖率={target:.2f}")
            print(f"最优传感器数量={len(refined_solution)}")
            print(f"实际覆盖率={coverage_ratio:.3f}")
            print(f"选择的传感器节点={refined_solution}")

        return sensor_counts, coverages, sensor_sets


if __name__ == "__main__":
    G = nx.random_graphs.erdos_renyi_graph(500, 0.02, seed=42)
    hsp = HybridSensorPlacement(G)

    thresholds = [1.0, 0.95, 0.9, 0.85, 0.8]
    sensor_counts, coverages = hsp.run_multi_threshold(thresholds)

    plt.figure(figsize=(8, 6))
    plt.plot(sensor_counts, coverages, marker="o")
    plt.xlabel("Sensor Count")
    plt.ylabel("Coverage Ratio")
    plt.title("Coverage vs Sensor Count")
    plt.grid()
    plt.show()
