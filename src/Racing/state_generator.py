import gym
import numpy as np
import random

def rgb2gray(rgb):
	return np.dot(rgb, [0.299, 0.587, 0.114])

def down_sample(state):
	state = rgb2gray(state)#self.rgb2gray(state[:82,:])
	return  state#downscale_local_mean(state, (2, 2)).flatten()

def get_random_states(num_samples):
	env = gym.make('CarRacing-v0')
	sample = []
	for episode in range(num_samples/5):
		states = []
		env.reset()
		done = False
		while not done:
			state, _, done, _ = env.step(env.action_space.sample())
			states.append(state)
		for state in random.sample(states, 5):
			sample.append(down_sample(state))
	return sample


sample = get_random_states(2000)

np.save('random_sample.npy', sample)