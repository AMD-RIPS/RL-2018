import sys
sys.dont_write_bytecode = True

import agent
import environment as env

environment = env.env_dict["CarRacing"]()
control = agent.DQN_Agent(environment=environment, architecture='atari', explore_rate='AtariPaper', learning_rate='basic')
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=100000)


control.train()
print(control.test_Q(10, visualize=True))
