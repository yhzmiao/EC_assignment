# DEAP
# Gaussian Mutation
# 2 point crossover


import random, math
import numpy as np

import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# setup the environment

experiment_name = "task1_sa"

n_hidden_neurons = 10
stage = 1 #1 3 8

env = Environment(experiment_name=experiment_name,
                  enemies=[1], # will be changed later
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  speed="fastest")

env.state_to_log()

n_variables = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 # number of values
#print(n_variables)  265   10 200 55 5

# setup the ea



#population
#chromosome
#gene

from deap import creator
from deap import base
from deap import tools
from deap import algorithms

IND_SIZE = n_variables # number of attributes
POP_SIZE = 100 # number of individuals

def calc_fitness(individual):
    # average of all enemies
    enermies_list = [5]
    fitness_list = []

    #print(individual)

    for e in enermies_list:
        env.update_parameter('enemies', [e])
        fitness_list.append(env.play(pcont=np.array(individual))[0])
    
    ret = np.mean(fitness_list)
    var = np.var(fitness_list)
    std = np.std(fitness_list)
    if (ret < 0):
        ret = min(2, 10 / (-ret))
    return ret, var, std

creator.create("FitnessMax", base.Fitness, weights=(100.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)



toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mutate", simple_mutate)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", calc_fitness)

def simulated_annealing(f1, f2, gen_id, T0):
    if (f1 < f2 + eps):
        print("Accepted!")
        return 0
    if (f1 > f2 and random.uniform(0, 1) < math.pow(gen_id, (1/f1 - 1/f2) / T0)):
        print("Accepted!")
        return 0
    print("Rejected!")
    return 1 # choose the old one

def uniform_mutate(individual):
    for i in range(len(individual)):
        individual[i] = random.uniform(-1, 1)
    return individual,



for iterator in range(4, 6):
    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = 1,

    np.random.seed(128)

    eps = 1e-8

    #algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 30, stats = stats, halloffame=hof)

    CX_RATE = 0.7
    MT_RATE = 0.3
    N_RESET = 10
    REPLACE_K = 30

    n_gen = 12 # number of generations

    gen_id = 2 # if 1, it will accept every change
    T0 = 0.1

    best_fitness = 0
    best_ind = None
    none_update_gen = 0
    updated_best = 0
    best_gen_fitness = 0
    best_gen_ind = None

    fp = open(experiment_name + "/output_5_" + str(iterator) + ".txt", "w")



    for now_gen in range(0, n_gen):
        print("Start %d generation!" % now_gen)
        offspring = toolbox.select(pop, len(pop))
        
        updated_best = 0
        best_gen_fitness = 0

        # crossover
        for i in range(1, len(offspring), 2):
            print("Now working with %d and %d individual!" % (i - 1, i))
            if random.uniform(0, 1) < CX_RATE:
                old_offspring_1 = toolbox.clone(offspring[i - 1])
                old_offspring_2 = toolbox.clone(offspring[i])
                old_offspring_1.fitness.values = toolbox.evaluate(old_offspring_1)
                old_offspring_2.fitness.values = toolbox.evaluate(old_offspring_2)

                # some best results might be dropped here
                if best_fitness + eps < old_offspring_1.fitness.values[0]:
                    best_fitness = old_offspring_1.fitness.values[0]
                    print(best_fitness)
                    best_ind = toolbox.clone(old_offspring_1)
                    updated_best = 1
                if best_gen_fitness + eps < old_offspring_1.fitness.values[0]:
                    best_gen_fitness = old_offspring_1.fitness.values[0]
                    print(best_gen_fitness)
                    best_gen_ind = toolbox.clone(old_offspring_1)

                if best_fitness + eps < old_offspring_2.fitness.values[0]:
                    best_fitness = old_offspring_2.fitness.values[0]
                    print(best_fitness)
                    best_ind = toolbox.clone(old_offspring_2)
                    updated_best = 1
                if best_gen_fitness + eps < old_offspring_2.fitness.values[0]:
                    best_gen_fitness = old_offspring_2.fitness.values[0]
                    print(best_gen_fitness)
                    best_gen_ind = toolbox.clone(old_offspring_2)
            
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                offspring[i - 1].fitness.values = toolbox.evaluate(offspring[i - 1])
                offspring[i].fitness.values = toolbox.evaluate(offspring[i])

                if(simulated_annealing(old_offspring_1.fitness.values[0], offspring[i - 1].fitness.values[0], gen_id, T0) == 1):
                    offspring[i - 1] = toolbox.clone(old_offspring_1)
                if(simulated_annealing(old_offspring_2.fitness.values[0], offspring[i].fitness.values[0], gen_id, T0) == 1):
                    offspring[i] = toolbox.clone(old_offspring_2)
            else:
                offspring[i - 1].fitness.values = toolbox.evaluate(offspring[i - 1])
                offspring[i].fitness.values = toolbox.evaluate(offspring[i])

        # simulated annealing(mutation)
        for i in range(0, len(offspring)):
            #offspring[i].fitness.values = toolbox.evaluate(offspring[i])
            if random.uniform(0, 1) > MT_RATE:
                continue
            old_offspring = toolbox.clone(offspring[i])

            
            # some best results might be dropped here
            if best_fitness + eps < old_offspring.fitness.values[0]:
                best_fitness = old_offspring.fitness.values[0]
                print(best_fitness)
                best_ind = toolbox.clone(old_offspring)
                updated_best = 1
            if best_gen_fitness + eps < old_offspring.fitness.values[0]:
                best_gen_fitness = old_offspring.fitness.values[0]
                print(best_gen_fitness)
                best_gen_ind = toolbox.clone(old_offspring)

            
            offspring[i], = uniform_mutate(offspring[i])
            offspring[i].fitness.values = toolbox.evaluate(offspring[i])

            # if not accept, return to old
            #print(old_offspring.fitness.values)
            if (simulated_annealing(old_offspring.fitness.values[0], offspring[i].fitness.values[0], gen_id, T0) == 1): # not accept
                offspring[i] = toolbox.clone(old_offspring)

        # replace k worst with k best
        fitness_list = []
        for ind in offspring:
            fitness_list.append(ind.fitness.values[0])
        rank_list = np.argsort(fitness_list)
        gen_mean = np.mean(fitness_list)
        gen_var = np.var(fitness_list)
        gen_std = np.std(fitness_list)
        for i in range(REPLACE_K):
            #print(offspring[rank_list[i]].fitness.values[0])
            offspring[rank_list[i]] = toolbox.clone(offspring[rank_list[len(offspring) - i - 1]])
            #print(offspring[rank_list[i]].fitness.values[0])
    
        # update best
        for ind in offspring:
            if best_fitness + eps < ind.fitness.values[0]:
                best_fitness = ind.fitness.values[0]
                print(best_fitness)
                best_ind = toolbox.clone(ind)
                updated_best = 1
            if best_gen_fitness + eps < ind.fitness.values[0]:
                best_gen_fitness = ind.fitness.values[0]
                print(best_gen_fitness)
                best_gen_ind = toolbox.clone(ind)

        none_update_gen += updated_best
    
        if none_update_gen >= N_RESET:
            none_update_gen = 0
            gen_id = 1
            T0 = T0 * 2 # start with more mutation
            print("New T0!")

        # output for making plot
        print("Best fitness in %d generation %f, Best fitness by now %f" % (now_gen, best_gen_ind.fitness.values[0], best_ind.fitness.values[0]), file = fp)
        print("This generation mean = %f, var = %f, std = %f" % (gen_mean, gen_var, gen_std), file = fp)

        # output to screen
        print("Best fitness in %d generation %f, Best fitness by now %f" % (now_gen, best_gen_ind.fitness.values[0], best_ind.fitness.values[0]))
        print("This generation mean = %f, var = %f, std = %f" % (gen_mean, gen_var, gen_std))
        print(iterator)

        gen_id += 1
        pop = offspring


    np.savetxt(experiment_name + '/best_5_' + str(iterator) + '.txt', best_ind)