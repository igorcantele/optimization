from deap import base, creator, tools
from .compare import compare
import numpy as np
import random
individuals = 10
import sys

if len(sys.argv) == 2:
    GENERATION = sys.argv[0]
    POPULATION = sys.argv[1]
elif len(sys.argv) == 1:
    GENERATION = sys.argv[0]
    POPULATION = 500
else:
    POPULATION = 500
    GENERATION = 100
def eval_err(individual):
    error = compare(individual[0], individual[1])
    return np.sum(error)

def custom_mutation(individual, indpb):
    return tools.mutGaussian(individual, 0, .1, indpb)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.random)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# genetic operators
toolbox.register("evaluate", eval_err)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutation, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    stats = []
    random.seed(64)

    # create an initial population of 300 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=POPULATION)

    # CXPB  is the probability with which two individuals are crossed
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = [fit]

    fits = [ind.fitness.values[0] for ind in pop]
    generation = 0

    while min(fits) > 0 and generation < GENERATION:
        print(f"Generation {generation}")
        # Select the next generation individuals
        old_offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, old_offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        stats.append(fits)
        best_ind = tools.selBest(pop, 1)[0]

        generation += 1
    return stats, best_ind

import pickle
def save_to_pickle(matrix, file):
    with open(file, "wb") as f:
        pickle.dump(matrix, f)

if __name__ == "__main__":
    from mpi4py.futures import MPIPoolExecutor
    pool = MPIPoolExecutor()
    toolbox.register("map", pool.map)
    fits, best= main()
    from pathlib import Path
    path = Path(__file__).parent.parent
    path_file = str(path) + "/prova_super.pkl"
    save_to_pickle({"fits": fits, "best":best}, path_file)
    pool.close()