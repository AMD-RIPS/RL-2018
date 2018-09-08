import sys
sys.dont_write_bytecode = True

import sys
import tensorflow as tf
import agent
import environment as env
import parameters.setup as setup

#####################################  Usage  ##########################################################
# 1) A command line argument specifying the name of the folder we want to log in must
#    be specified when this file is run, like so: "python main.py name_of_folder".
# 2) The parameters for DQN_Agent and CarRacing are defined in the setup_dict object 
#    in parameters/setup.py.
########################################################################################################

environment = env.CarRacing(**setup.setup_dict['car racing'])
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[1], **setup.setup_dict['agent'])

#####################################  Traning a model  ################################################
# control.train(test_frequency = 30, test_episodes = 5)

#####################################  Testing a model  ################################################
##### 
control.load("./models/best_model/data.chkp-900")
print('Average score: {0} +- {1}'.format(*control.test(num_test_episodes=100, visualize=True, log=True)))