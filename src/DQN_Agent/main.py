import sys
sys.dont_write_bytecode = True

import sys
import agent
import environment as env
import tensorflow as tf

environment = env.env_dict["Pong"]()
control = agent.DQN_Agent(environment=environment, architecture='atari', explore_rate='atari', learning_rate='atari', model_name=sys.argv[1])
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=1000, num_episodes=1000)
control.train()
# control.load('/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/tmp/data.chkp')
print(control.test_Q(5, visualize=True))
