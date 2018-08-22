import sys
sys.dont_write_bytecode = True

import sys
import agent
import environment as env
import tensorflow as tf
import parameters.setup as setup

environment = env.CarRacing(**setup.setup_dict['car racing'])
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[1], **setup.setup_dict['agent'])

# Traning a model
control.train()

# Testing a model
# control.load("/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/tmp/data.chkp-1")
# control.test_Q(5, True)