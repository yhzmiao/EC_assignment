import random, math
import numpy as np

import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

# setup the environment

experiment_name = "task1"

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

n_gen = 50 # number of generations
mut_rate = 0.075 # mutation rate

#population
#chromosome
#gene

from deap import creator
from deap import base
from deap import tools
from deap import cma
from deap import algorithms

IND_SIZE = n_variables # number of attributes
POP_SIZE = 100 # number of individuals

def calc_fitness(individual):
    # average of all enemies
    enermies_list = [1, 2, 3, 4, 5, 6, 7, 8]
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
    sigma = 0.6
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


toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", simple_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", calc_fitness)

algorithms.eaSimple(pop, toolbox, 0.7, 0.1, 30, stats = stats, halloffame=hof)

print(calc_fitness(hof[0]))
np.savetxt(experiment_name+'/best.txt', hof[0])