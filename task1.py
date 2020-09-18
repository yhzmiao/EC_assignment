import random
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

# setup the ea

n_gen = 50 # number of generations
n_pop = 100 # number of populations
mut_rate = 0.075 # mutation rate

#population
#chromosome
#gene


#initial population


def init_population(type_pop):
    if (type_pop == 0):
        population = np.array([[random.uniform(-1, 1) for j in range(n_variables)] for i in range(n_pop)])
        return population
    if (type_pop == 1):
        population = np.array([[0.0 for j in range(n_variables)] for i in range(n_pop)])
        return population



def calc_fitness(chromosome):
    return env.play(pcont=chromosome)[0]


def dup(new_pop, new_pop_id, now_pop, now_pop_id):
    for i in range(n_variables):
        new_pop[new_pop_id][i] = now_pop[now_pop_id][i]

def get_random_id(sum_fit, fitness_array):
    target_fit = random.uniform(0, sum_fit)
    #print(target_fit)
    now_sum_fit = 0
    i = 0
    while now_sum_fit < target_fit:
        now_sum_fit += fitness_array[i]
        i = i + 1
    return i - 1


def next_gen(gen_id, now_enermies, now_pop, new_pop, best, select, crossover, mutation):
    fitness = np.array([[0.0 for j in range(8)] for i in range(n_pop)])
    fitness_mean = np.array([0.0 for i in range(n_pop)])

    #if (gen_id == 0):
    enermies = [1, 2, 3, 4, 5, 6, 7, 8]
        #now_enermies = enermies
    #else:
    #    enermies = now_enermies
    
    # task = [i for i in range ] # running 1 3 8 
    for i in range(n_pop):
        print("For %dth gen %dth chromo" % (gen_id, i))
        for j in range(len(enermies)):
            env.update_parameter('enemies', [enermies[j]])
            fitness[i][j] = calc_fitness(now_pop[i]) # this part can be run mutithread
        fitness_mean[i] = np.mean(fitness[i,0:len(enermies)])
        if (fitness_mean[i] < 0):
            fitness_mean[i] = min(1.0, 2.0 / (-fitness_mean[i]))
    
    # the best 15
    print(fitness_mean)
    rank = np.argsort(-fitness_mean)
        
    for i in range(best):
        dup(new_pop, i, now_pop, rank[i])

    sum_fit = np.sum(fitness_mean)
    # select 25 by their value
    for i in range(select):
        rand_id = get_random_id(sum_fit, fitness_mean)
        #print(rand_id)
        dup(new_pop, i + best, now_pop, rand_id - 1)

    #crossover 58
    for i in range(crossover):
        rand_id_1 = get_random_id(sum_fit, fitness_mean)
        rand_id_2 = get_random_id(sum_fit, fitness_mean)
        crossover_point = random.randint(10, n_variables - 10)
        for j in range(n_variables):
            if (j < crossover_point):
                new_pop[i * 2 + best + select][j] = now_pop[rand_id_1][j]
                new_pop[i * 2 + 1 + best + select][j] = now_pop[rand_id_2][j]
            else:
                new_pop[i * 2 + best + select][j] = now_pop[rand_id_2][j]
                new_pop[i * 2 + 1 + best + select][j] = now_pop[rand_id_1][j]
    #mutation 2
    for i in range(mutation):
        rand_id = get_random_id(sum_fit, fitness_mean)
        dup(new_pop, n_pop - i - 1, now_pop, rand_id)
        for j in range(n_variables):
            if (random.uniform(0, 1) < mut_rate):
                new_pop[n_pop - i - 1][j] = random.uniform(-1, 1)
        
    
    enermies_rate = [0.8 for i in range(8)]
    #if gen_id > 10
    #for i in range(len(now_enermies)):
    #    if (fitness[rank[0]][i] > 80):
    #        enermies_rate[now_enermies[i] - 1] = 0.45
    #    elif (fitness[rank[0]][i] > 60):
    #        enermies_rate[now_enermies[i] - 1] = 0.9
    #    elif (fitness[rank[0]][i] < 10):
    #        enermies_rate[now_enermies[i] - 1] = 0.7
    #    else:
    #        enermies_rate[now_enermies[i] - 1] = 0.6
    #print(enermies_rate)
    #enermies = []
    #for i in range(8):
    #    if(random.uniform(0, 1) < enermies_rate[i]):
    #        enermies.append(i + 1)

    return rank[0], fitness_mean[rank[0]], np.var(fitness[rank[0], 0:len(enermies)]), np.std(fitness[rank[0], 0:len(enermies)]), enermies




now_pop = init_population(0)
new_pop = init_population(1)

best_gen = 0
best_mean = 0

f = open("out.txt", "w")

now_enermies = []

for i in range(n_gen):
    print("Generation:%d" % i)
    if i - best_gen > 10:
        break

    # 15 best   25 select   58 crossover   2 mutation  

    now_best_id, now_best_mean, var, std, now_enermies= next_gen(i, now_enermies, now_pop, new_pop, 15, 25, 58 // 2, 2)
    print("Best result of Gen %d is %d: avg = %.8f, var = %.8f, std = %.8f"%(i, now_best_id, now_best_mean, var, std), file = f)
    print(now_enermies)

    if (now_best_mean > best_mean):
        best_gen = i
        best_mean = now_best_mean
        np.savetxt(experiment_name+'/best.txt',now_pop[now_best_id])
    now_pop = new_pop

