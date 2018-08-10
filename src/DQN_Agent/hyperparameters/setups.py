CartPole = {
	'general': {
		'architecture': 'basic', 
		'explore_rate': 'basic', 
		'learning_rate': 'basic'
	}, 

	'training': {
		'target_update_frequency': 200,
		'discount': .99, 
		'batch_size': 32, 
		'memory_capacity': 1000, 
		'num_episodes': 10000,
		'score_limit': 199, 
		'replay_method': 'prioritized'
	}
}

Pong = {
	'general': {
		'architecture': 'nature', 
		'explore_rate': 'decay', 
		'learning_rate': 'atari'
	}, 

	'training': {
		'target_update_frequency': 1000,
		'discount': .99, 
		'batch_size': 32, 
		'memory_capacity': 100000, 
		'num_episodes': 10000,
		'score_limit': 19, 
		'replay_method': 'prioritized'
	}
}

BreakOut = {
	'general': {
		'architecture': 'atari', 
		'explore_rate': 'atari', 
		'learning_rate': 'atari'
	},

	'training': {
		'target_update_frequency': 1000,
		'discount': .99, 
		'batch_size': 32, 
		'memory_capacity': 100000, 
		'num_episodes': 10000,
		'score_limit': 300, 
		'replay_method': 'regular'
	}
}

CarRacing = {
	'general': {
		'architecture': 'nature', 
		'explore_rate': 'decay', 
		'learning_rate': 'atari'
	},

	'training': {
		'target_update_frequency': 1000,
		'batch_size': 32, 
		'memory_capacity': 100000, 
		'num_episodes': 9000,
		'learning_rate_drop_frame_limit': 250000
	}
}

setup_dict = {
	'CartPole': CartPole,
	'Pong': Pong,
	'BreakOut': BreakOut,
	'CarRacing': CarRacing
}