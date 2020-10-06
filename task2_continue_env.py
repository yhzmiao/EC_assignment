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

experiment_name = "task2_continue_env"
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
    e_list = [1, 3, 5 ,7, 2, 4, 6, 8]
    env = Environment(experiment_name=experiment_name,
                    enemies=[e_list[e_id]], # will be changed later
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons),
                    enemymode="static",
                    speed="fastest")
    play_result = env.play(pcont=np.array(ind))
    
    fitness_list[e_id] = play_result[0]
    gain_list[e_id] = play_result[1] - play_result[2]

def calc_fitness(individual):
    list_size = 4

    fitness_list = Array('d', range(list_size))
    gain_list = Array('d', range(list_size))

    #print(individual)
    #e_list = [1, 3, 5 ,7, 2, 4, 6, 8]

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

def uniform_mutate(individual):
    for i in range(len(individual)):
        individual[i] = random.uniform(-1, 1)
    return individual,


def simulated_annealing(f1, f2, gen_id, T0):
    if (f1 < f2 + eps):
        print("Accepted!")
        return 0
    if (f1 > f2 and random.uniform(0, 1) < math.pow(gen_id, (1/f1 - 1/f2) / T0)):
        print("Accepted!")
        return 0
    print("Rejected!")
    return 1 # choose the old one

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
    
    # continue with best results
    f_cont = open("cont_best.txt", "r")
    for i in range(n_variables):
        v = f_cont.readline()
        pop[0][i] = float(v)

    #print(pop[0])
    for i in range(1, POP_SIZE):
        if i % 4 == 0:
            pop[i] = toolbox.clone(pop[0])
    f_cont.close()
    #print(pop)

    
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
                #print(old_offspring_1.fitness.values)
                old_offspring_2.fitness.values = toolbox.evaluate(old_offspring_2)

                # some best results might be dropped here
                if best_fitness + eps < old_offspring_1.fitness.values[0]:
                    best_fitness = old_offspring_1.fitness.values[0]
                    best_ind = toolbox.clone(old_offspring_1)
                    updated_best = 1
                if best_gen_fitness + eps < old_offspring_1.fitness.values[0]:
                    best_gen_fitness = old_offspring_1.fitness.values[0]
                    best_gen_ind = toolbox.clone(old_offspring_1)

                if best_fitness + eps < old_offspring_2.fitness.values[0]:
                    best_fitness = old_offspring_2.fitness.values[0]
                    best_ind = toolbox.clone(old_offspring_2)
                    updated_best = 1
                if best_gen_fitness + eps < old_offspring_2.fitness.values[0]:
                    best_gen_fitness = old_offspring_2.fitness.values[0]
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
        gain_list = []
        for ind in offspring:
            fitness_list.append(ind.fitness.values[0])
            gain_list.append(ind.fitness.values[3])
        rank_list = np.argsort(fitness_list)
        gen_mean = np.mean(fitness_list)
        gen_gain_mean = np.mean(gain_list)
        gen_var = np.var(fitness_list)
        gen_gain_var = np.var(gain_list)
        gen_std = np.std(fitness_list)
        gen_gain_std = np.std(gain_list)
        
        if (now_gen >= 10):
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