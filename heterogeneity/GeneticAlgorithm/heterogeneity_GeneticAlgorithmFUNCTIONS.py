import numpy as np
import time
from heterogeneity.GeneticAlgorithm.heterogeneity_JansenRitDavidOPTIMIZEE import fitness_JRD

def get_selection_probabilities(selection_strategy, pop_keep):

    if selection_strategy == "roulette_wheel":

        mating_prob = (np.arange(1, pop_keep + 1) / np.arange(1, pop_keep + 1).sum())[::-1]

        return np.array([0, *np.cumsum(mating_prob[: pop_keep + 1])])

    elif selection_strategy == "random":
        return np.linspace(0, 1, pop_keep + 1)


def initialize_population(pop_size, variables, n_rois):
    """
    Initializes the population of the problem according to the
    population size and number of genes (variables).
    :param pop_size: number of individuals in the population
    :param variables: tuple containing the minimum and maximum allowed
    :return: a numpy array with a randomly initialized population
    """

    population = list()
    for var, lim in variables.items():
        if "heterogeneity" in var:
            population.append(np.random.uniform(lim[0], lim[1], size=(pop_size, n_rois)))

        else:
            population.append(np.random.uniform(lim[0], lim[1], size=(pop_size, 1)))

    population = np.concatenate(population, axis=1)

    return population


def calculate_fitness(population, optimizee_params, variables, n_rois, verbose=False):
    """
    Calculates fitness for each individual in population.
    :param population: group of individuals
    :param optimizee_params: parameters
    :return:
    """
    if optimizee_params["mode"] == "FC":
        fit = list()

        for i, individual in enumerate(population):
            tic = time.time()
            fit.append(fitness_JRD(individual, optimizee_params, variables, verbose))
            if verbose:
                print("Individual %i took %0.2fs" % (i, time.time()-tic,))
        return np.array(fit)

    elif optimizee_params["mode"] == "FFT":
        fit_pre = list()
        w_updated = list()
        fit_post = list()
        for i, individual in enumerate(population):
            tic = time.time()
            temp = fitness_JRD(individual, optimizee_params, variables, verbose)
            fit_pre.append(temp[1])
            w_updated.append(temp[0])
            fit_post.append(temp[2])
            if verbose:
                print("Individual %i took %0.2fs" % (i, time.time()-tic,))

        return np.array(fit_post), np.asarray(w_updated), np.array(fit_pre)


def sort_by_fitness(fitness, population):
    """
    Sorts the population by its fitness.
    :param fitness: fitness of the population
    :param population: population state at a given iteration
    :return: the sorted fitness array and sorted population array
    """

    sorted_fitness = np.argsort(fitness)[::-1]

    population = population[sorted_fitness, :]
    fitness = fitness[sorted_fitness]

    return fitness, population


def select_parents(selection_strategy, n_matings, prob_intervals):
    """
    Selects the parents according to a given selection strategy.
    Options are:
    roulette_wheel: Selects individuals from mating pool giving
    higher probabilities to fitter individuals.

    :param selection_strategy: the strategy to use for selecting parents
    :param n_matings: the number of matings to perform
    :param prob_intervals: the selection probability for each individual in
     the mating pool.
    :return: 2 arrays with selected individuals corresponding to each parent
    """

    ma, pa = None, None

    if selection_strategy == "roulette_wheel":
        ma = np.apply_along_axis(
            lambda value: np.argmin(value > prob_intervals) - 1, 1, np.random.rand(n_matings, 1)
        )
        pa = np.apply_along_axis(
            lambda value: np.argmin(value > prob_intervals) - 1, 1, np.random.rand(n_matings, 1)
        )

    else:
        print('This selection strategy was not developed. Go to select_parents function and add it.')

    return ma, pa


def create_offspring(first_parent, sec_parent, crossover_pt, offspring_number, variables, n_rois):
    """
    Creates an offspring from 2 parents. It performs the crossover
    according the following rule:
    p_new = first_parent[crossover_pt] + beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
    offspring = [first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:]
    where beta is a random number between 0 and 1, and can be either positive or negative
    depending on if it's the first or second offspring
    :param first_parent: first parent's chromosome
    :param sec_parent: second parent's chromosome
    :param crossover_pt: point(s) at which to perform the crossover
    :param offspring_number: whether it's the first or second offspring from a pair of parents.

    :return: the resulting offspring.
    """

    beta = (
        np.random.rand(1)[0]
        if offspring_number == "first"
        else -np.random.rand(1)[0]
    )

    p_new = first_parent[crossover_pt] - beta * (
            first_parent[crossover_pt] - sec_parent[crossover_pt]
    )

    # To what variable corresponds this crossover point?
    var_list = []
    i = 0
    for var in variables.keys():
        if "heterogeneity" in var:
            for roi in range(n_rois):
                var_list.append(i)
            i = i + 1
        else:
            var_list.append(i)
            i = i + 1



    if p_new < list(variables.values())[var_list[int(crossover_pt)]][0]:  # We dont expect crossovers to get under the limits
        p_new = list(variables.values())[var_list[int(crossover_pt)]][0]

    if p_new > list(variables.values())[var_list[int(crossover_pt)]][1]:  # We dont expect crossovers to get over the limits
        p_new = list(variables.values())[var_list[int(crossover_pt)]][1]

    return np.hstack(
        (first_parent[:int(crossover_pt)], p_new, sec_parent[int(crossover_pt) + 1:]))


def mutate_population(population, n_mutations, pop_size, variables, n_rois):
    """
    Mutates the population by randomizing specific positions of the
    population individuals.
    :param population: the population at a given iteration
    :param n_mutations: number of mutations to be performed.
    :param input_limits: tuple containing the minimum and maximum allowed
     values of the problem space.

    :return: the mutated population
    """

    mutation_rows = np.random.choice(
        np.arange(1, population.shape[0]), n_mutations, replace=True
    )

    mutation_columns = np.random.choice(
        population.shape[1], n_mutations, replace=True
    )

    new_population = initialize_population(pop_size, variables, n_rois)

    population[mutation_rows, mutation_columns] = new_population[mutation_rows, mutation_columns]

    return population