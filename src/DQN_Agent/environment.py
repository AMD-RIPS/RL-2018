import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from skimage.transform import downscale_local_mean
from PIL import Image


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")



class CartPole:

    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.state_space_size = np.shape(self.env.observation_space)[0]
        self.state_space_lower_bounds = self.env.observation_space.low
        self.state_space_upper_bounds = self.env.observation_space.high
        self.action_space_size = self.env.action_space.n
        self.history = []
        self.state_shape = [None,4]

    def sample_action_space(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    # Returns a (4, ) vector
    def process(self, x):
        return x

    def add_history(self, state, action, reward):
        pass

class Pong:

    def __init__(self):
        self.image_dim = 42*32
        self.env = gym.make("Pong-v0")
        self.state_space_size = 4*self.image_dim
        # self.state_space_lower_bounds = self.env.observation_space.low
        # self.state_space_upper_bounds = self.env.observation_space.high
        self.action_space_size = self.env.action_space.n
        self.history = []
        self.history_pick = 4
        self.state_shape = [None, self.history_pick, 42, 32]

    def sample_action_space(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()
    
    # Returns 42*32 greyscale image
    def downscale(self, rgb):
    	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = downscale_local_mean(gray, (5, 5))
        # im = Image.fromarray(gray)
        # im.show()
        # print(gray.shape)
        # pause()
    	return gray

    def process(self, state):
    	if len(self.history) < 3: 
            result = np.zeros(shape=(1, 4*self.image_dim)).flatten()
            return result
    	result = np.empty(shape=(1,0))
    	for image in self.history:
    		temp = self.downscale(image).reshape([1, self.image_dim])
    		result = np.concatenate((result, temp), axis = 1)
    	result = np.concatenate((result, self.downscale(state).reshape([1, self.image_dim])), axis = 1).flatten()
        return result

    def add_history(self, state, action, reward):
        if len(self.history) < self.history_pick - 1: 
            zeros = np.zeros((42,32))
            result = [zeros, zeros, zeros, zeros]
            return result
        result = []
        for image in self.history:
            temp = self.downscale(image)#.reshape([1, self.image_dim])
            result.append(temp)
        result.append(self.downscale(state))
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick - 1: self.history.pop(0)
        self.history.append(state)

class CarRacing:

    def __init__(self):
        self.image_dim = 48*48
        self.env = gym.make("CarRacing-v0")
        self.action_space_size = 4
        self.history = []
        self.history_pick = 4
        self.state_space_size = self.image_dim * self.history_pick 
        self.state_shape = [None, self.history_pick, 48, 48]

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action_index):
        return [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.8]][action_index]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(self.map_action(action))

    def render(self):
        self.env.render()

    def downscale(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = downscale_local_mean(gray, (2, 2))
        return gray

    def process(self, state):
        if len(self.history) < self.history_pick - 1: 
            zeros = np.zeros((48,48))
            result = [zeros, zeros, zeros, zeros]
            return result
        result = []
        for image in self.history:
            temp = self.downscale(image)#.reshape([1, self.image_dim])
            result.append(temp)
        result.append(self.downscale(state))
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick - 1: self.history.pop(0)
        self.history.append(state)

env_dict = {
	"CartPole": CartPole,
	"Pong": Pong,
    "CarRacing": CarRacing
}
