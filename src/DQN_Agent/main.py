import sys
sys.dont_write_bytecode = True

import agent
import environment as env

# Train Cartpole
environment = env.env_dict["BreakOut"]()
control = agent.DQN_Agent(environment=environment, architecture='conv2', explore_rate='basic', learning_rate='basic')
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=300)


control.train()
print(control.test_Q(10, visualize=True))
