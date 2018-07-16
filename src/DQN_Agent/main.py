import sys
sys.dont_write_bytecode = True

import agent
import environment as env

# Train Cartpole
environment = env.Environment(game='CartPole-v0')
cp = agent.DQN_Agent(environment=environment, architecture='basic', explore_rate='basic', learning_rate='basic')
cp.set_training_parameters(discount=.99, batch_size=16, memory_capacity=10000, num_episodes=300)
cp.train()
print(cp.test_Q(10, visualize=True))
