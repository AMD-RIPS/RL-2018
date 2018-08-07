import sys
sys.dont_write_bytecode = True

import sys
import agent
import environment as env
import tensorflow as tf
import hyperparameters.setups as setups

game = 'CarRacing'

training_environment = env.env_dict[game]()
testing_environment = env.env_dict[game](test=True)
control = agent.DQN_Agent(training_environment=training_environment, testing_environment=testing_environment, model_name=sys.argv[1], **setups.setup_dict[game]['general'])
control.set_training_parameters(**setups.setup_dict[game]['training'])
control.load("/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/penalised/data.chkp-361")
# control.train()
print(control.test_Q(10, visualize=True))

