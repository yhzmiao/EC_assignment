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

experiment_name = "task1_deap_2"
# experiment_name = "task1_deap"

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
    enermies_list = [3]
    fitness_list = []

    #print(individual)

    for e in enermies_list:
        env.update_parameter('enemies', [e])
        fitness_list.append(env.play(pcont=np.array(individual))[0])
    
    ret = np.mean(fitness_list)
    if (ret < 0):
        ret = min(2, 10 / (-ret))
    return ret,

creator.create("FitnessMax", base.Fitness, weights=(100.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=POP_SIZE)

np.random.seed(128)

hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

def simple_mutate(individual):
    sigma = 0.2
    eps = 0.1
    new_sigma = sigma * math.exp(0.1 * np.random.normal(loc = 0.0, scale = 1.0, size = None))
    new_sigma = max(new_sigma, eps)
    for i in range(len(individual)):
        individual[i] += new_sigma * np.random.normal(loc = 0.0, scale = 1.0, size = None)
        if individual[i] < -1:      # boundary
            individual[i] = -1
        if individual[i] > 1:
            individual[i] = 1
    return individual,

def self_adaptive_mutate(individual, sigma):
    for i in range(len(individual)):
        individual[i] += sigma * np.random.normal(loc = 0.0, scale = 1.0, size = None)
        if individual[i] < -1:      # boundary
            individual[i] = -1
        if individual[i] > 1:
            individual[i] = 1
    return individual,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", simple_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", calc_fitness)

CX_RATE = 0.7
MT_RATE = 0.1
REPLACE_K = 25
n_gen = 20
eps = 1e-8

MAX_NON_UPDATE = 10
none_update_gen = 0
end = 0
best_fitness = 0
best_ind = None
best_gen_fitness = 0
best_gen_ind = None

# use the build in method of deap
#algorithms.eaSimple(pop, toolbox, CX_RATE, MT_RATE, n_gen, stats = stats, halloffame=hof)

# rewrite eaSimple

fp = open(experiment_name + "/output.txt", "w")

for now_gen in range(n_gen):
    print("Start %d generation!" % now_gen)
    offspring = toolbox.select(pop, len(pop))

    # select
    for i in range(1, len(offspring), 2):
        if random.uniform(0, 1) < CX_RATE:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])

    # mutate
    sigma_mutation = 0.2
    sigma_eps = 0.1
    for i in range(len(offspring)):
        print("Now for %d:" % i)
        if random.uniform(0, 1) < CX_RATE:
            # simple mutation
            offspring[i], = simple_mutate(offspring[i])
            # self adaptive mutation
            # sigma_mutation = max(sigma_mutation * math.exp(0.1 * np.random.normal(loc = 0.0, scale = 1.0, size = None)), sigma_eps)
            # offspring[i], = self_adaptive_mutate(offspring[i], sigma_mutation)
        offspring[i].fitness.values = toolbox.evaluate(offspring[i])
        print(offspring[i].fitness.values)
    
    # replace k worst with k best   
    fitness_list = []
    for ind in offspring:
        fitness_list.append(ind.fitness.values[0])
    rank_list = np.argsort(fitness_list)
    # values for this gen
    gen_mean = np.mean(fitness_list)
    gen_var = np.var(fitness_list)
    gen_std = np.std(fitness_list)
    for i in range(REPLACE_K):
        offspring[rank_list[i]] = toolbox.clone(offspring[rank_list[len(offspring) - i - 1]])

    # record the best 
    updated_best = 1
    best_gen_fitness = 0
    for ind in offspring:
        if best_fitness + eps < ind.fitness.values[0]:
            best_fitness = ind.fitness.values[0]
            print(best_fitness)
            best_ind = toolbox.clone(ind)
            updated_best = 0
            none_update_gen = 0
        if best_gen_fitness + eps < ind.fitness.values[0]:
            best_gen_fitness = ind.fitness.values[0]
            print(best_gen_fitness)
            best_gen_ind = toolbox.clone(ind)

    none_update_gen += updated_best
    
    # end if 10 gen no update
    if none_update_gen >= MAX_NON_UPDATE:
        none_update_gen = 0
        end = 1

    # output for making plot
    print("Best fitness in %d generation %f, Best fitness by now %f" % (now_gen, best_gen_ind.fitness.values[0], best_ind.fitness.values[0]), file = fp)
    print("This generation mean = %f, var = %f, std = %f" % (gen_mean, gen_var, gen_std), file = fp)

    # output to screen
    print("Best fitness in %d generation %f, Best fitness by now %f" % (now_gen, best_gen_ind.fitness.values[0], best_ind.fitness.values[0]))
    print("This generation mean = %f, var = %f, std = %f" % (gen_mean, gen_var, gen_std))
    pop = offspring

    #if end == 1:
    #    break

# for DEAP
#print(calc_fitness(hof[0]))
#np.savetxt(experiment_name+'/best.txt', hof[0])

# or
np.savetxt(experiment_name+'/best.txt', best_ind)