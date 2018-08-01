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
		'memory_capacity': 10000, 
		'num_episodes': 10000,
		'replay_method': 'regular'
	}
}

Pong = {
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
		'num_episodes': 3000,
		'replay_method': 'regular'
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
		'replay_method': 'regular'
	}
}

CarRacing = {
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
		'replay_method': 'regular'
	}
}

setup_dict = {
	'CartPole': CartPole,
	'Pong': Pong,
	'BreakOut': BreakOut,
	'CarRacing': CarRacing
}