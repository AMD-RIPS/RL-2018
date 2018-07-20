import sys
sys.dont_write_bytecode = True

import agent
import environment as env

# Train Breakout
<<<<<<< 9854b7d4d9d413790865d68c355178700b6610f9
environment = env.env_dict["BreakOut"]()
=======
environment = env.env_dict["Pong"]()
>>>>>>> Merge conflicts resolved, working Breakout version
control = agent.DQN_Agent(environment=environment, architecture='conv2', explore_rate='AtariPaper', learning_rate='basic')
control.set_training_parameters(discount=.99, batch_size=32, memory_capacity=10000, num_episodes=100000)


control.train()
print(control.test_Q(10, visualize=True))
