import sys
sys.dont_write_bytecode = True

import sys
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
control = agent.DQN_Agent(training_environment=training_environment, testing_environment=testing_environment, model_name=sys.argv[1], **setups.setup_dict['general'])
# control.load("/home/pgerber/Documents/RL-2018/src/DQN_Agent/models/flip/data.chkp-1441")
control.set_training_parameters(**setups.setup_dict['training'])
control.train()
# control.test_Q(5, True)