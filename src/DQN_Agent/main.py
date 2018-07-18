import sys
sys.dont_write_bytecode = True

import agent
import environment as env

# Train Cartpole
environment = env.env_dict["CarRacing"]()
control = agent.DQN_Agent(environment=environment, architecture='conv2', explore_rate='basic', learning_rate='basic', skip_frames=4)
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=1)

control.train()
print(control.test_Q(10, visualize=True))
