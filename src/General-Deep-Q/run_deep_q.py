import deep_q_general

# Train Cartpole
print ('---- CartPole ----')
cp = deep_q_general.Playground('CartPole-v1', 2, [24,24], .4, .01, .05, .99, 32, 10000)
cp.begin_training(1500)

# Train MountainCar
# print ('---- Mountain Car ----')
# mc = deep_q_general.Playground('MountainCar-v0', 2, [24,24], 1, .01, .05, .99, 32, 10000)
# mc.begin_training(10000)

# Train Acrobot
# print ('---- Acrobot ----')
# acro = deep_q_general.Playground('Acrobot-v1', 2, [24,24], 1, .01, .05, .99, 32, 10000)
# acro.begin_training(10000)
