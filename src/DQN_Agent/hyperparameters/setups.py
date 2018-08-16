setup_dict = {
	'general': {
		'architecture': 'nature', 
		'explore_rate': 'decay', 
		'learning_rate': 'atari'
	},

	'training': {
		'target_update_frequency': 1000,
		'batch_size': 32, 
		'memory_capacity': 100000, 
		'num_episodes': 3000,
		'learning_rate_drop_frame_limit': 250000
	}
}