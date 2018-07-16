import sys
sys.dont_write_bytecode = True

import agent

# Train Cartpole
cp = agent.DQN_Agent(game = 'CartPole-v0', architecture = 'basic', explore_rate = 'basic', learning_rate = 'basic', discount = .99, batch_size = 16, memory_capacity = 10000)
cp.train(300)
print(cp.test_Q(10, visualize = True))