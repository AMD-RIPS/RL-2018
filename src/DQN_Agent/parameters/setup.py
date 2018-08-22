fixed_1track_seed = [108]
fixed_3track_seed = [104, 106, 108]

from explore_rates import *
from learning_rates import * 
from architectures import *

setup_dict = {
	'agent': {
		'architecture': Nature_Paper_Conv_Dropout, 
		'learning_rate': Atari_Learning_Rate,
		'explore_rate': Decay_Explore_Rate,
		'target_update_frequency': 1000,
		'batch_size': 32, 
		'memory_capacity': 100000, 
		'num_episodes': 3000,
		'learning_rate_drop_frame_limit': 250000
	},

	'car racing': {
		'seed': fixed_3track_seed, 
		'detect_edges': False, 
		'detect_grass': False, 
		'flip': True,
		'type': 'ShortTrack'
	}
}

