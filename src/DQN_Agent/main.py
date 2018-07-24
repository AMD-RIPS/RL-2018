import sys
sys.dont_write_bytecode = True

import agent
import environment as env
import tensorflow as tf

environment = env.env_dict["CarRacing"](history_pick=4)
control = agent.DQN_Agent(environment=environment, architecture='atari', explore_rate='atari', learning_rate='atari')
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=100000)
# control.load("/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/Pong_1532450521.47/data.chkp")
control.train()
control.test_Q(10, visualize=True)
