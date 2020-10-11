# DEAP
# Gaussian Mutation
# 2 point crossover


import random, math
from multiprocessing import Process, Array
import numpy as np

import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

os.environ["SDL_VIDEODRIVER"] = "dummy"

# setup the environment

experiment_name = "task2_self_2"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10
    #env[e - 1].state_to_log()

n_variables = 265#(env[0].get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 # number of values
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

NUM_OF_CORE = 4 # 2 4 8 would be optional

def multi_play(e_id, fitness_list, gain_list, ind):
    e_list = [4, 6, 7, 8]
    env = Environment(experiment_name=experiment_name,
                    enemies=[e_list[e_id]], # will be changed later
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    speed="fastest")
    play_result = env.play(pcont=np.array(ind))
    
    fitness_list[e_id] = play_result[0]
    gain_list[e_id] = play_result[1] - play_result[2]

def calc_fitness(individual, n_gen):
    list_size = 4

    fitness_list = Array('d', range(list_size))
    gain_list = Array('d', range(list_size))

    #print(individual)
    #e_list = [1, 3, 5 ,7, 2, 4, 6, 8]

    if list_size == 2:
        p0 = Process(target = multi_play, args = (0, fitness_list, gain_list, individual))
        p1 = Process(target = multi_play, args = (1, fitness_list, gain_list, individual))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
    else:
        for e in range(0, list_size, NUM_OF_CORE):
            process_list = []
            for i in range(NUM_OF_CORE):
                process_list.append(Process(target = multi_play, args = (e + i, fitness_list, gain_list, individual)))
            for i in range(NUM_OF_CORE):
                process_list[i].start()
            for i in range(NUM_OF_CORE):
                process_list[i].join()
    
    ret = np.mean(fitness_list)
    var = np.var(fitness_list)
    std = np.std(fitness_list)
    gain = np.sum(gain_list)
    #print(gain_list)
    if (ret < 0):
        ret = min(2, 10 / (-ret))
    print(ret, var, std, gain)
    return (ret, var, std, gain)



#np.random.seed(128)
eps = 1e-8

def self_adaptive_mutate(individual, sigma):
    for i in range(len(individual)):
        individual[i] += sigma * np.random.normal(loc = 0.0, scale = 1.0, size = None)
        if individual[i] < -1:      # boundary
            individual[i] = -1
        if individual[i] > 1:
            individual[i] = 1
    return individual,

creator.create("FitnessMax", base.Fitness, weights=(100.0, 10000.0, 100.0, 800.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=POP_SIZE)
for ind in pop:
    ind.fitness.values = 1.0, 0.0, 0.0, 0.0

toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mutate", simple_mutate)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", calc_fitness)

if __name__ == "__main__":
    
    n_gen = 30 # number of generations

    gen_id = 2 # if 1, it will accept every change
    T0 = 0.1

    CX_RATE = 0.5
    MT_RATE = 0.2
    N_RESET = 10
    REPLACE_K = 30

    best_fitness = 0
    best_ind = None
    none_update_gen = 0
    updated_best = 0
    best_gen_fitness = 0
    best_gen_ind = None
    
    for now_gen in range(0, 60):
        if now_gen == 30:
            best_fitness = 0
        print("Start %d generation!" % now_gen)
        offspring = toolbox.select(pop, len(pop))
        
        updated_best = 0
        best_gen_fitness = 0

        # crossover
        for i in range(1, len(offspring), 2):
            if random.uniform(0, 1) < CX_RATE:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])

        # mutation
        sigma_mutation = 0.2
        sigma_eps = 0.1
        for i in range(0, len(offspring)):
            #offspring[i].fitness.values = toolbox.evaluate(offspring[i])
            if random.uniform(0, 1) < MT_RATE:
                # self adaptive mutation
                sigma_mutation = max(sigma_mutation * math.exp(0.1 * np.random.normal(loc = 0.0, scale = 1.0, size = None)), sigma_eps)
                offspring[i], = self_adaptive_mutate(offspring[i], sigma_mutation)

        # replace k worst with k best
        fitness_list = []
        gain_list = []
        for i in range(0, len(offspring)):
            print("Now working with %d individual!" % (i))
            offspring[i].fitness.values = toolbox.evaluate(offspring[i], now_gen)
            fitness_list.append(offspring[i].fitness.values[0])
            gain_list.append(offspring[i].fitness.values[3])
        rank_list = np.argsort(fitness_list)
        gen_mean = np.mean(fitness_list)
        gen_gain_mean = np.mean(gain_list)
        gen_var = np.var(fitness_list)
        gen_gain_var = np.var(gain_list)
        gen_std = np.std(fitness_list)
        gen_gain_std = np.std(gain_list)
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
        
        fp = open(experiment_name + "/output.txt", "a")
        # output for making plot
        print("Best fitness in %d generation %f, gain %f, best fitness by now %f, gain %f" % (now_gen, best_gen_ind.fitness.values[0], best_gen_ind.fitness.values[3], best_ind.fitness.values[0], best_ind.fitness.values[3]), file = fp)
        print("This generation fitness mean = %f, var = %f, std = %f" % (gen_mean, gen_var, gen_std), file = fp)
        print("This generation gain mean = %f, var = %f, std = %f" % (gen_gain_mean, gen_gain_var, gen_gain_std), file = fp)
        fp.close()

        # output to screen
        print("Best fitness in %d generation %f, gain %f, best fitness by now %f, gain %f" % (now_gen, best_gen_ind.fitness.values[0], best_gen_ind.fitness.values[3], best_ind.fitness.values[0], best_ind.fitness.values[3]))
        print("This generation fitness mean = %f, var = %f, std = %f" % (gen_mean, gen_var, gen_std))
        print("This generation gain mean = %f, var = %f, std = %f" % (gen_gain_mean, gen_gain_var, gen_gain_std))

        gen_id += 1
        pop = offspring


    np.savetxt(experiment_name+'/best.txt', best_ind)
    fp_end = open(experiment_name + "/end.txt", "w")
    for ind in pop:
        print(ind, file = fp_end)
    fp_end.close()