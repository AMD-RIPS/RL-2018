import sys
sys.dont_write_bytecode = True

import agent
import environment as env

# Train Cartpole
environment = env.env_dict["CarRacing"]()
control = agent.DQN_Agent(environment=environment, architecture='conv2', explore_rate='basic', learning_rate='basic')
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=300)
control.train()
print(control.test_Q(10, visualize=True))

# Train Pong
# environment = env.env_dict["Pong"]()
# control = agent.DQN_Agent(environment=environment, architecture='conv', explore_rate='basic', learning_rate='basic')
# control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=500)
# control.train()
# print(control.test_Q(10, visualize=True))
