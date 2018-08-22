import sys
sys.dont_write_bytecode = True

import os
import agent
import environment as env
import tensorflow as tf
import hyperparameters.setups as setups

fixed_1track_seed = [108]
fixed_3track_seed = [104, 106, 108]

# One curve
# training_environment = env.env_dict[game](type='OneCurve', detect_edges=False)

# Two curve
# training_environment = env.env_dict[game](type='ShortTrack')

# Fixed full track
# training_environment = env.env_dict[game](seed=fixed_3track_seed)

flip = True
detect_edges = False
detect_grass = False
training_environment = env.CarRacing(seed=fixed_3track_seed, detect_edges=detect_edges, detect_grass=detect_grass, flip=flip)

testing_environment = env.CarRacing(test=True, detect_edges=detect_edges, detect_grass=detect_grass, flip=False)
control = agent.DQN_Agent(training_environment=training_environment, testing_environment=testing_environment, model_name='whatever', **setups.setup_dict['general'])

DIR = './models/flip'
model = sys.argv[1]
num_test_episodes = 2
number_of_checkpoints = (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 2) / 3
for n in range(number_of_checkpoints):
	chkp = (n + 1)*30 + 1
	control.load("./models/{0}/data.chkp-{1}".format(model, chkp))
	control.set_training_parameters(**setups.setup_dict['training'])
	print('Currently testing checkpoint {0}'.format(chkp))
	mean, std, rewards = control.test_Q(num_test_episodes, True)
	print('Checkpoint {0} got a score of {1} +- {2}'.format(chkp, mean, std))
	file = open('./models/{0}/rewards.txt'.format(model),'a')
	file.write(','.join(map(str, rewards)) + '\n')
	file.close()