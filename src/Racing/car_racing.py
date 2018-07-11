import gym
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean

env = gym.make('CarRacing-v0')


s = env.reset()
env.render()


def rgb2gray(rgb):
    return np.dot(rgb, [0.299, 0.587, 0.114])

def down_sample(state):
	state = rgb2gray(state[:82,:])
	return  downscale_local_mean(state, (2, 2)).flatten()

def get_state_space_size(state):
	return np.shape(down_sample(state))[0]

# print env.action_space.sample()

steering = [-1, -.5, 0, .5, 1]
acceleration = [0, .5, 1]
decceleration = [0, .5, 1]
a_len = len(acceleration)
d_len = len(decceleration)
def get_action(action_index):
	s = steering[int(np.floor(action_index/(a_len*d_len)))]
	a = acceleration[int(np.floor(action_index/d_len))%a_len]
	d = decceleration[action_index%d_len]
	return [s,a,d]

for i in range(45):
	print get_action(i)