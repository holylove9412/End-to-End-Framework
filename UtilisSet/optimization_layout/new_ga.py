import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SensorPlacementGA:
    def __init__(self, graph, population_size=50, generations=200, crossover_prob=0.8, mutation_prob=0.05, sensor_upbound=None):
        self.graph = graph
        self.node_num = len(graph.nodes)
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.sensor_upbound = sensor_upbound if sensor_upbound else self.node_num
        self.K = 10
        self.khop_neighbors = {
            n: set(nx.single_source_shortest_path_length(graph, n, cutoff=self.K).keys())
            for n in graph.nodes
        }

    # def _init_population(self):
    #     return np.Random.randint(0, 2, (self.population_size, self.node_num))

    def _fitness(self, chromosome):
        selected = np.where(chromosome == 1)[0]
        covered_nodes = set()
        for s in selected:
            covered_nodes |= self.khop_neighbors[s]

        coverage_ratio = len(covered_nodes) / self.node_num
        sensor_ratio = len(selected) / self.node_num

        # 惩罚未覆盖的节点
        return sensor_ratio + 10 * (1 - coverage_ratio)

    def _selection(self, population, fitness):
        idx1, idx2 = np.random.randint(0, self.population_size, 2)
        return population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

    # def _crossover(self, parent1, parent2):
    #     if np.Random.rand() < self.crossover_prob:
    #         point = np.Random.randint(1, self.node_num - 1)
    #         child1 = np.concatenate((parent1[:point], parent2[point:]))
    #         child2 = np.concatenate((parent2[:point], parent1[point:]))
    #         return child1, child2
    #     return parent1.copy(), parent2.copy()

    # def _mutation(self, chromosome):
    #     for i in range(self.node_num):
    #         if np.Random.rand() < self.mutation_prob:
    #             chromosome[i] = 1 - chromosome[i]
    #     return chromosome

    def _init_population(self):
        population = np.zeros((self.population_size, self.node_num), dtype=int)
        for i in range(self.population_size):
            ones = np.random.choice(self.node_num, size=np.random.randint(1, self.sensor_upbound + 1), replace=False)
            population[i, ones] = 1
        return population

    def _repair(self, chromosome):
        ones_idx = np.where(chromosome == 1)[0]
        if len(ones_idx) > self.sensor_upbound:
            np.random.shuffle(ones_idx)
            chromosome[ones_idx[self.sensor_upbound:]] = 0
        return chromosome

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            point = np.random.randint(1, self.node_num - 1)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        return self._repair(child1), self._repair(child2)

    def _mutation(self, chromosome):
        for i in range(self.node_num):
            if np.random.rand() < self.mutation_prob:
                chromosome[i] = 1 - chromosome[i]
        return self._repair(chromosome)

    def run(self, sensor_upbound=None):
        if sensor_upbound:
            self.sensor_upbound = sensor_upbound

        population = self._init_population()
        best_fitness_list = []

        for gen in range(self.generations):
            fitness = np.array([self._fitness(ind) for ind in population])
            sorted_idx = np.argsort(fitness)
            best_individual = population[sorted_idx[0]].copy()
            best_fitness = fitness[sorted_idx[0]]
            best_fitness_list.append(best_fitness)

            new_population = [best_individual]

            while len(new_population) < self.population_size:
                parent1 = self._selection(population, fitness)
                parent2 = self._selection(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)

                if np.sum(child1) > self.sensor_upbound:
                    ones_idx = np.where(child1 == 1)[0]
                    np.random.shuffle(ones_idx)
                    child1[ones_idx[self.sensor_upbound:]] = 0
                if np.sum(child2) > self.sensor_upbound:
                    ones_idx = np.where(child2 == 1)[0]
                    np.random.shuffle(ones_idx)
                    child2[ones_idx[self.sensor_upbound:]] = 0

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population)
        final_fitness = np.array([self._fitness(ind) for ind in population])
        best_idx = np.argmin(final_fitness)
        best_solution = population[best_idx]
        selected_sensors = np.where(best_solution == 1)[0]

        covered = set()
        for s in selected_sensors:
            covered |= self.khop_neighbors[s]
        coverage_ratio = len(covered) / self.node_num

        return selected_sensors, coverage_ratio, best_fitness_list

def run_multi_upperbound(graph, start, step=5):
    ga = SensorPlacementGA(graph, generations=5000, population_size=80)
    sensor_counts = []
    coverages = []
    sensor_sets = []

    for ub in range(start, 0, -step):
        sensors, coverage, _ = ga.run(sensor_upbound=ub)
        sensor_counts.append(len(sensors))
        coverages.append(coverage)
        sensor_sets.append(sensors)
        print(f"Sensor Upbound={ub} | Selected={len(sensors)} | Coverage={coverage:.3f}")

    return sensor_counts, coverages, sensor_sets
