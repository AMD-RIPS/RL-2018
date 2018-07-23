import sys
sys.dont_write_bytecode = True

import agent_init_reload
import environment as env

# Train Breakout
environment = env.env_dict["Classic_Control"]('CartPole-v0')
control = agent_init_reload.DQN_Agent(environment=environment, architecture='basic', 
	explore_rate='atari', learning_rate='basic', reload_bool=False)
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=10000)
control.train()
print(control.test_Q(10, visualize=True))