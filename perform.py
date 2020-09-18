import numpy as np

import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

n_hidden_neurons = 10

env = Environment(experiment_name="perform",
                  enemies=[1], # will be changed later
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  speed="normal")

env.state_to_log()

n_variables = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
n_var = np.array([0.0 for i in range(n_variables)])

with open("best.txt", 'r') as fp:
	for i in range(n_variables):
		line = fp.readline()
		if not line:
			break
		n_var[i] = float(line)

print(n_var)

for enermy in range(8):
	env.update_parameter('enemies', [enermy + 1])
	env.play(pcont=n_var)
