import sys
sys.dont_write_bytecode = True

import sys
import agent
import environment as env
import tensorflow as tf
import parameters.setup as setup

environment = env.CarRacing(**setup.setup_dict['car racing'])
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[1], **setup.setup_dict['agent'])

DIR = './models/flip'
model = sys.argv[1]
save_frequency = 30
num_test_episodes = 2
number_of_checkpoints = (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 2) / 3
for n in range(number_of_checkpoints):
	chkp = (n + 1)*save_frequency + 1
	control.load("./models/{0}/data.chkp-{1}".format(model, chkp))
	print('Currently testing checkpoint {0}'.format(chkp))
	mean, std, rewards = control.test_Q(num_test_episodes, True)
	print('Checkpoint {0} got a score of {1} +- {2}'.format(chkp, mean, std))
	file = open('./models/{0}/rewards.txt'.format(model),'a')
	file.write(','.join(map(str, rewards)) + '\n')
	file.close()