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
control.train()

#####################################  Testing a model  ################################################
##### 
# control.load("/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/tmp/data.chkp-1")
# control.test_Q(5, True)