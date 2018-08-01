import sys
sys.dont_write_bytecode = True

import sys
import agent
import environment as env
import tensorflow as tf

environment = env.env_dict["CarRacing"](type = "OneCurve", seed = 2)
control = agent.DQN_Agent(environment=environment, architecture='nature', explore_rate='decay', learning_rate='atari', model_name=sys.argv[1])
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=10000)
# control.load("/home/jguan/Documents/RIPS/RL-2018/src/DQN_Agent/models/carracing_oneturn_1000frames_accumulatereward_blacktile")
control.train()
print(control.test_Q(5, visualize=True))
