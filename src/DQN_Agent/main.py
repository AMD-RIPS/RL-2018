import sys
sys.dont_write_bytecode = True

import agent
import environment as env
import tensorflow as tf

environment = env.env_dict["Pong"]()
control = agent.DQN_Agent(environment=environment, architecture='conv2', explore_rate='atari', learning_rate='atari')
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=100000, num_episodes=100000, save = True)
# control.load("/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/Pong_1532556300.18/data.chkp")
control.train()
print(control.test_Q(5, visualize=True))
