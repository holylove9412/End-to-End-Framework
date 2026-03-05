import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

class SensorPlacementGA1:
    def __init__(self, graph, population_size=50, generations=100, crossover_prob=0.8, mutation_prob=0.05, elite_fraction=0.05,sensor_upbound=None):
        self.graph = graph
        self.elite_fraction = elite_fraction
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

    def _coverage(self, chromosome):
        selected = np.where(chromosome == 1)[0]
        covered_nodes = set()
        for s in selected:
            covered_nodes |= self.khop_neighbors[s]
        return len(covered_nodes) / self.node_num

    def _selection(self, population, fitness, k=3):
        candidates = np.random.choice(self.population_size, k, replace=False)
        best_idx = candidates[np.argmin(fitness[candidates])]
        return population[best_idx]

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
            sorted_idx = np.argsort(fitness)
            elite_num = max(1, int(self.elite_fraction * self.population_size))
            elite_individuals = population[sorted_idx[:elite_num]]
            elite_fitness = fitness[sorted_idx[:elite_num]]

            best_fitness = elite_fitness[0]
            best_fitness_list.append(best_fitness)

            new_population = list(elite_individuals)
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

            selected = np.where(elite_individuals[0] == 1)[0]
            covered_nodes = set()
            for s in selected:
                covered_nodes |= self.khop_neighbors[s]
            coverage_ratio = len(covered_nodes) / self.node_num

            print(
                f"Generation {gen + 1}/{self.generations} | "
                f"Best Fitness: {best_fitness:.4f} | "
                f"Sensors: {len(selected)} | "
                f"Coverage: {coverage_ratio:.3f}"
            )

        final_fitness = np.array([self._fitness(ind) for ind in population])
        best_idx = np.argmin(final_fitness)
        best_solution = population[best_idx]
        selected_sensors = np.where(best_solution == 1)[0]

        return selected_sensors, best_fitness_list

    def visualize(self, best_solution):
        # --- 图1：传感器数量变化 ---
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.best_sensor_count_list, '-o', label='Sensor Count', color='orange')
        plt.xlabel("Generation")
        plt.ylabel("Number of Sensors")
        plt.title("Best Sensor Count Evolution")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.best_coverage_list, '-o', label='Coverage Ratio', color='blue')
        plt.xlabel("Generation")
        plt.ylabel("Coverage Ratio")
        plt.title("Best Coverage Evolution")
        plt.legend()
        plt.tight_layout()
        plt.show()

        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3)
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=[i for i in self.graph.nodes if best_solution[i] == 0],
                               node_color="lightblue", label="Unselected")
        nx.draw_networkx_nodes(self.graph, pos,
                               nodelist=[i for i in self.graph.nodes if best_solution[i] == 1],
                               node_color="red", label="Sensor")
        plt.title("Final Sensor Layout")
        plt.legend()
        plt.show()
