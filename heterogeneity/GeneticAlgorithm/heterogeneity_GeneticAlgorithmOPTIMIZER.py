import math

import pandas as pd

from tvb.simulator.lab import connectivity
from heterogeneity.GeneticAlgorithm.heterogeneity_GeneticAlgorithmFUNCTIONS import *

f = open("GA_results/r_log"+time.strftime("%d%b_%H.%M", time.gmtime())+".txt", 'w')

tic0 = time.time()

# Indexes ROI for DMN in AAL2red structure: Frontal_Sup; ACC; PCC; Parietal_inf; Precuneus; Temporal_Mid.
AAL2red_rois_dmn = [2, 3, 34, 35, 38, 39, 64, 65, 70, 71, 84, 85]

################################################ PARAMETERS
## Optimizer parameters
pop_size = 30  # population size [40]
max_gen_round1, max_gen_round2 = 20, 10  # Maximum number of GA generations per round. 'None' for just a round. [50, 50]


selection_strategy = "roulette_wheel"  # Uses individual fitness to calculate probability of selection
selection_rate = 0.5
mutation_rate = 0.25

## Optimizee global parameters
optimizee_params = {
    "mode": "FC",  # FC, FFT. Whenever FFT gradient descent will be applied.
    "GD_iterations": 15,
    "GD_learning_rate": 2,

    "simLength": 14000,
    "transient": 2000,
    "samplingFreq": 1000,
    "reps": 3,

    "emp_subj": "NEMOS_035",
    "structure": "_AAL2red",
    "sc_subset": AAL2red_rois_dmn,  # Subsets over structure
    "fc_subset": AAL2red_rois_dmn,  # Subsets over AAL2red (main FC output shape from brainstorm).
    "ctb_folder": "D:\\Users\Jesus CabreraAlvarez\PycharmProjects\\brainModels\\CTB_data\\output\\",
    "verbose": True}

## VARIABLES to optimize (genes): g, s, p, sigma, ¿w? and variable limits {var: [low, high]}
## If you want to free any variable ROIwise use: "var_name + _heterogeneity".
variables = {"g": [0, 250], "s": [0.5, 40], "p": [0, 0.22], "sigma": [0, 0.022]}

############################################################### end

# Configuring Genetic Algorithm: setting up parameters
conn = connectivity.Connectivity.from_file(optimizee_params["ctb_folder"] + optimizee_params["emp_subj"] + optimizee_params["structure"]+".zip")
if optimizee_params["sc_subset"]:
    n_rois = len(optimizee_params["sc_subset"])
else:
    n_rois = len(conn.region_labels)

n_genes = sum([n_rois if "heterogeneity" in var else 1 for var in variables.keys()])

## Write down parameters
f.write("ROUND 1 paramters")
f.write("\npopulation size: " + str(pop_size))
f.write("\ngenerations r1: " + str(max_gen_round1))
f.write("\ngenerations r2: " + str(max_gen_round2))
f.write("\nselection strategy: " + str(selection_strategy))
f.write("\nselection rate: " + str(selection_rate))
f.write("\nmutation rate: " + str(mutation_rate))

f.write("\n\ngradient descent iteration: " + str(optimizee_params["GD_iterations"]))
f.write("\ngradient descent learning rate: " + str(optimizee_params["GD_learning_rate"]))

f.write("\n\nsimulation length: " + str(optimizee_params["simLength"]))
f.write("\nsubject: " + optimizee_params["emp_subj"])
f.write("\nstructure: " + optimizee_params["structure"])
f.write("\nsc subset: " + str(optimizee_params["sc_subset"]))
f.write("\nn rois: " + str(n_rois))

for key, vals in variables.items():
    f.write("\n" + key + ": [" + str(vals[0]) + ", " + str(vals[1]) + "]")
###


if len(variables) == n_genes and optimizee_params["mode"] == "FFT":
    print("Wait, what variable are you going to optimize for FFT?")
    quit()

pop_keep = math.floor(selection_rate * pop_size)  # number of individuals to keep on each iteration

n_matings = math.floor((pop_size - pop_keep) / 2)  # number of crossovers to perform
n_mutations = math.ceil((pop_size - 1) * len(variables) * mutation_rate)  # number of mutations to perform

# probability intervals, needed for roulette_wheel and random selection strategies
prob_intervals = get_selection_probabilities(selection_strategy, pop_keep)

Results = list()
print("Initializing Genetic Algorithm  .  Wait, its coming.")

# initialize population randomly
population = initialize_population(pop_size, variables, n_rois)

# Calculate the fitness of the population
if optimizee_params["mode"] == "FC":
    fitness = calculate_fitness(population, optimizee_params, variables, n_rois, verbose=optimizee_params["verbose"])
elif optimizee_params["mode"] == "FFT":
    fitness, w_updated = calculate_fitness(population, optimizee_params, variables, n_rois, verbose=optimizee_params["verbose"])
    population[:, 4:] = w_updated  # update w in population

# Sort population by fitness
fitness, population = sort_by_fitness(fitness, population)

print(" Generation nº 0 |  Best performance params  |  Fitness (FC)  . time: %0.2f  min" % ((time.time()-tic0)/60,))
print(population[0], fitness[0])

gen_n = 0
while gen_n < max_gen_round1:

    tic = time.time()
    gen_n += 1

    # Get parents pairs
    ma, pa = select_parents(selection_strategy, n_matings, prob_intervals)

    # Get indices of individuals to be replaced
    ix = np.arange(0, pop_size - pop_keep - 1, 2)

    # Get crossover point for each individual
    xp = np.random.randint(0, n_genes, size=(pop_size, 1))

    for i in range(ix.shape[0]):
        # create first offspring
        population[-1 - ix[i], :] = create_offspring(
            population[ma[i], :], population[pa[i], :], xp[i], "first", variables, n_rois
        )

        # create second offspring
        population[-1 - ix[i] - 1, :] = create_offspring(
            population[pa[i], :], population[ma[i], :], xp[i], "second", variables, n_rois
        )

    population = mutate_population(population, n_mutations, pop_size, variables, n_rois)

    # Get new population's fitness. Since the fittest element does not change,
    # we do not need to re calculate its fitness
    if optimizee_params["mode"] == "FC":
        fitness = np.hstack((fitness[0], calculate_fitness(population[1:, :], optimizee_params, variables, n_rois)))

    elif optimizee_params["mode"] == "FFT":
        fitness_temp, w_updated = calculate_fitness(population, optimizee_params, variables, n_rois, verbose=False)
        fitness = np.hstack((fitness[0], fitness_temp))
        population[1:, 4:] = w_updated  # update w in population

    fitness, population = sort_by_fitness(fitness, population)

    print(" Generation nº %i |  Best performance params  |  Fitness (FC)  . time: %0.2f / %0.2f  min" % (gen_n, (time.time()-tic)/60, (time.time()-tic0)/60))
    print(population[0], fitness[0])

    Results.append([list(population[0]) + [fitness[0]]])

dfResults = pd.DataFrame(np.squeeze(np.asarray(Results)), columns=["g", "s", "p", "sigma", "rFC"])
dfPopulation = np.concatenate((population, np.expand_dims(fitness, axis=1)), axis=1)
dfPopulation = pd.DataFrame(np.asarray(dfPopulation), columns=["g", "s", "p", "sigma", "rFC"])

dfResults.to_csv("GA_results/r1_fitness_evolution"+time.strftime("%d%b_%H.%M", time.gmtime())+'-tFit_'+str(round((time.time()-tic0)/60, 2))+"m.csv", index=False)
dfPopulation.to_csv("GA_results/r1_survivors"+time.strftime("%d%b_%H.%M", time.gmtime())+'-tFit_'+str(round((time.time()-tic0)/60, 2))+"m.csv", index=False)


gen_n = 0
Results_r2 = list()
if max_gen_round2:  # at a certain number of rounds, switch to "FFT" mode. Optimizing then good individuals.

    print("\n\n\n\n  ROUND TWO: fft fitting")

    optimizee_params["mode"] = "FFT"
    optimizee_params["verbose"] = False
    variables["w_heterogeneity"] = [0, 1]
    n_genes = sum([n_rois if "heterogeneity" in var else 1 for var in variables.keys()])

    # Transition from population with 4 genes to population with 4+n_rois genes
    population_r2 = np.ones((len(population), n_genes)) * 0.8
    population_r2[:, :len(population[0])] = population

    while gen_n < max_gen_round2:

        tic = time.time()
        gen_n += 1

        # Get parents pairs
        ma, pa = select_parents(selection_strategy, n_matings, prob_intervals)

        # Get indices of individuals to be replaced
        ix = np.arange(0, pop_size - pop_keep - 1, 2)

        # Get crossover point for each individual
        xp = np.random.randint(0, n_genes, size=(pop_size, 1))

        for i in range(ix.shape[0]):
            # create first offspring
            population_r2[-1 - ix[i], :] = create_offspring(
                population_r2[ma[i], :], population_r2[pa[i], :], xp[i], "first", variables, n_rois
            )

            # create second offspring
            population_r2[-1 - ix[i] - 1, :] = create_offspring(
                population_r2[pa[i], :], population_r2[ma[i], :], xp[i], "second", variables, n_rois
            )

        population_r2 = mutate_population(population_r2, n_mutations, pop_size, variables, n_rois)

        # Get new population's fitness. Since the fittest element does not change,
        # we do not need to re calculate its fitness
        if optimizee_params["mode"] == "FC":
            fitness = np.hstack((fitness[0], calculate_fitness(population_r2[1:, :], optimizee_params, variables, n_rois)))

        elif optimizee_params["mode"] == "FFT":
            fitness_post_temp, w_updated, fitness_pre_temp = calculate_fitness(population_r2[1:, :], optimizee_params, variables, n_rois, verbose=optimizee_params["verbose"])
            fitness = np.hstack((fitness[0], fitness_post_temp))
            fitness_pre = np.hstack((fitness[0], fitness_pre_temp))
            population_r2[1:, 4:] = w_updated  # update w in population

        fitness, population_r2 = sort_by_fitness(fitness, population_r2)

        print(" Generation nº %i |  Best performance params  |  Fitness (FC) [post | pre]  . time: %0.2f / %0.2f  min" % (gen_n, (time.time()-tic)/60, (time.time()-tic0)/60))
        print(population_r2[0], fitness[0], fitness_pre[0])

        Results_r2.append([list(population_r2[0]) + [fitness[0]]])


dfResults = pd.DataFrame(np.squeeze(np.asarray(Results_r2)))  # saves the evolution of the best performer
dfPopulation = np.concatenate((population_r2, np.expand_dims(fitness, axis=1)), axis=1)
dfPopulation = pd.DataFrame(np.asarray(dfPopulation))  # saves the final set of survivors

dfResults.to_csv("GA_results/r2_fitness_evolution"+time.strftime("%d%b_%H.%M", time.gmtime())+'-tFit_'+str(round((time.time()-tic0)/60, 2))+"m.csv", index=False)
dfPopulation.to_csv("GA_results/r2_survivors"+time.strftime("%d%b_%H.%M", time.gmtime())+'-tFit_'+str(round((time.time()-tic0)/60, 2))+"m.csv", index=False)


f.write("\\\\ROUND 2 parameters")
f.write("\npopulation size: " + str(pop_size))
f.write("\ngenerations r1: " + str(max_gen_round1))
f.write("\ngenerations r2: " + str(max_gen_round2))
f.write("\nselection strategy: " + str(selection_strategy))
f.write("\nselection rate: " + str(selection_rate))
f.write("\nmutation rate: " + str(mutation_rate))

f.write("\n\ngradient descent iteration: " + str(optimizee_params["GD_iterations"]))
f.write("\ngradient descent learning rate: " + str(optimizee_params["GD_learning_rate"]))

f.write("\n\nsimulation length: " + str(optimizee_params["simLength"]))
f.write("\nsubject: " + optimizee_params["emp_subj"])
f.write("\nstructure: " + optimizee_params["structure"])
f.write("\nsc subset: " + str(optimizee_params["sc_subset"]))
f.write("\nn rois: " + str(n_rois))

for key, vals in variables.items():
    f.write("\n" + key + ": [" + str(vals[0]) + ", " + str(vals[1]) + "]")

f.close()


