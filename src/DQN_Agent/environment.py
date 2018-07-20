import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from skimage.transform import downscale_local_mean
from PIL import Image


def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")

# image standardization method
def per_image_standardization(image):
    image  = np.array(image)
    return (image - np.mean(image))/np.std(image)

class CartPole:

    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.state_space_size = np.shape(self.env.observation_space)[0]
        self.action_space_size = self.env.action_space.n
        self.history = []
        self.state_shape = [None, self.state_space_size]
        self.skip_frames = 1

    def sample_action_space(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        for i in range(self.skip_frames):
            next_state, reward, done, _ = self.env.step(action)
            if done:
                break
        info = {'true_done': done}
        return next_state, reward, done, info
    def render(self):
        self.env.render()

    # Returns a (sample_space_size, ) vector
    def process(self, x):
        return x

    def add_history(self, state, action, reward):
        pass

class Pong:

    def __init__(self):
        self.image_dim = 28*27
        self.env = gym.make("PongDeterministic-v4")
        self.state_space_size = 4*self.image_dim
        self.action_space_size = 2
        self.history = []
        self.history_pick = 4
        self.state_shape = [None, self.history_pick, 42, 32]
        self.skip_frames = 1

    def sample_action_space(self):
        return random.sample([0, 1, 2], 1)[0]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action_dict = {0:2, 1:3, 2:0}
        action = action_dict[action]
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            info = {'true_done': done}
            if reward == -1: done = True
            if done:
                break
        return next_state, reward, done, info

    def render(self):
        self.env.render()
    
    # Returns 28*27 greyscale image
    def downscale(self, rgb):
        rgb = rgb[33:196,:,:]
    	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    	gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = downscale_local_mean(gray, (6, 6))
        gray = per_image_standardization(gray)
        # im = Image.fromarray(gray)
        # im.show()
        # print(gray.shape)
        # pause()
    	return gray

    def process(self, state):
    	self.add_history(state, None, None)
        if len(self.history) < self.history_pick : 
            zeros = np.zeros((28,27))
            result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
            result = np.concatenate((result,np.array(self.history)))
        else: 
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick: self.history.pop(0)
        self.history.append(self.downscale(state))

class CarRacing:

    def __init__(self):
        self.image_dim = 48*48
        self.env = gym.make("CarRacing-v0")
        self.action_space_size = 4
        self.history = []
        self.history_pick = 4
        self.state_space_size = self.image_dim * self.history_pick 
        self.state_shape = [None, self.history_pick, 48, 48]
        self.skip_frame = 4

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action_index):
        return [[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.8]][action_index]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(self.map_action(action))
            if done:
                break
        info = {'true_done': done}
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def downscale(self, rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = downscale_local_mean(gray, (2, 2))
        gray = per_image_standardization(gray)
        return gray

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick : 
            zeros = np.zeros((48,48))
            result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
            result = np.concatenate((result,np.array(self.history)))
        else: 
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick: self.history.pop(0)
        self.history.append(self.downscale(state))

class BreakOut:

    def __init__(self):
        self.image_dim = 80*72
        self.env = gym.make("BreakoutDeterministic-v4")
        self.action_space_size = self.env.action_space.n
        self.history = []
        self.history_pick = 4
        self.state_space_size = self.image_dim * self.history_pick 
        self.state_shape = [None, self.history_pick, 80, 72]
        self.skip_frames = 1

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action_index):
        return action_index

    def reset(self):
        return self.env.reset()

    def step(self, action):
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(self.map_action(action))
            if done:
                break
        info = {'true_done': done}
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def downscale(self, rgb):
        rgb = rgb[34:-16, 8:-8, :]
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        gray = downscale_local_mean(gray, (2, 2))
        gray = per_image_standardization(gray)
        return gray

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick : 
            zeros = np.zeros((80,72))
            result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
            result = np.concatenate((result,np.array(self.history)))
        else: 
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick: self.history.pop(0)
        self.history.append(self.downscale(state))



env_dict = {
    "CartPole": CartPole,
    "Pong": Pong,
    "CarRacing": CarRacing,
    "BreakOut": BreakOut,
    "MountainCar": MountainCar
 }

