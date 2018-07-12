import deep_q_general

# Train Cartpole
print ('---- CartPole ----')
cp = deep_q_general.Playground(game = 'CartPole-v0', num_hidden_layers = 2, 
	layer_sizes = [16,16], epsilon_max = .4, epsilon_min = .01, 
	alpha = .005, gamma = .99, batch_size = 16, memory_capacity = 10000)
cp.begin_training(1000)
print(cp.test_Q(10, visualize = True))

# Train MountainCar
# print ('---- Mountain Car ----')
# mc = deep_q_general.Playground('MountainCar-v0', 2, [24,24], 1, .01, .05, .99, 32, 10000)
# mc.begin_training(10000)

# Train Acrobot
# print ('---- Acrobot ----')
# acro = deep_q_general.Playground('Acrobot-v1', 2, [24,24], 1, .01, .05, .99, 32, 10000)
# acro.begin_training(10000)
