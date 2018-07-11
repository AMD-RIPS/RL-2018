import deep_q_general
import time
# Train Cartpole
print ('---- CartPole ----')
start_time = time.time()
cp = deep_q_general.Playground('CartPole-v1', 2, [24,24], .4, .01, .05, .99, 32, 10000)
cp.begin_training(500)
print("--- %s seconds ---" % (time.time() - start_time))
cp.test_Q(100)
# GPU: --- 178.743749857 seconds ---
# CPU: --- 152.234107971 seconds ---

# Train MountainCar
# print ('---- Mountain Car ----')
# mc = deep_q_general.Playground('MountainCar-v0', 2, [24,24], 1, .01, .05, .99, 32, 10000)
# mc.begin_training(10000)

# Train Acrobot
# print ('---- Acrobot ----')
# acro = deep_q_general.Playground('Acrobot-v1', 2, [24,24], 1, .01, .05, .99, 32, 10000)
# acro.begin_training(10000)
