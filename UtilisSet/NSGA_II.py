import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

def evaluate_individual(individual, voi_vector, te_matrix):
    indices = np.where(np.array(individual) == 1)[0]
    voi = np.sum(voi_vector[indices])
    te = np.sum([te_matrix[i, j] for i in indices for j in indices if i < j])
    cost = len(indices)
    return -voi, te, cost
def run_nsga2(voi_vector, te_matrix, max_sensors, n_gen=100, pop_size=100):
    n = len(voi_vector)

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: np.random.choice([0, 1]))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def valid_individual(ind):
        while sum(ind) > max_sensors:
            i = np.random.choice(np.where(np.array(ind) == 1)[0])
            ind[i] = 0
        return ind

    def evaluate(ind):
        ind = valid_individual(ind)
        return evaluate_individual(ind, voi_vector, te_matrix)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.3,
                              ngen=n_gen, verbose=True)

    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pareto_front

def generate_mock_data(n=50):
    np.random.seed(42)
    voi_vector = np.random.rand(n)
    te_matrix = np.random.rand(n, n)
    te_matrix = (te_matrix + te_matrix.T) / 2
    np.fill_diagonal(te_matrix, 0)
    return voi_vector, te_matrix

def plot_pareto_front(solutions):
    vois, tes, costs = [], [], []
    for ind in solutions:
        v, t, c = ind.fitness.values
        vois.append(-v)
        tes.append(t)
        costs.append(c)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vois, tes, costs, c='blue', marker='o')
    ax.set_xlabel('VOI')
    ax.set_ylabel('TE')
    ax.set_zlabel('Sensor Count')
    ax.set_title('Pareto Front - NSGA-II')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    voi_vector, te_matrix = generate_mock_data(n=50)
    pareto_solutions = run_nsga2(voi_vector, te_matrix, max_sensors=10, n_gen=50, pop_size=60)
    plot_pareto_front(pareto_solutions)

    for ind in pareto_solutions:
        print("目标值 (VOI, TE, k):", ind.fitness.values)
        print("布设节点索引:", np.where(np.array(ind) == 1)[0])
