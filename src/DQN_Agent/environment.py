import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils

class Classic_Control:
    def __init__(self, game):
        self.env = gym.make(game)
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
        next_state, reward, done, _ = self.env.step(action)
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

    def __init__(self, crop = (34, -16, 8, -8), downscaling_factor = (2, 2), history_pick = 1, skip_frames = 1):
        self.env = gym.make('PongDeterministic-v4')
        self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
        self.history_pick = history_pick
        self.state_space_size = history_pick*np.prod(self.image_shape)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.image_shape)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0:2, 1:3, 2:0}
        self.crop = crop
        self.downscaling_factor = downscaling_factor

    def map_action(self, action):
        return self.action_dict[action]

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = self.map_action(action)
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            info = {'true_done': done}
            if done: break
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick : 
            zeros = np.zeros(self.image_shape)
            result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
            result = np.concatenate((result,np.array(self.history)))
        else: 
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick: self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))

class CarRacing:

    def __init__(self, crop = (None, None, None, None), downscaling_factor = (2, 2), history_pick = 1, skip_frames = 4):
        self.env = gym.make('CarRacing-v0')
        self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
        self.history_pick = history_pick
        self.state_space_size = history_pick*np.prod(self.image_shape)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.image_shape)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0: [-1, 0, 0],1: [1, 0, 0],2: [0, 1, 0],3: [0, 0, 0.8]}
        self.crop = crop
        self.downscaling_factor = downscaling_factor

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_dict[action]

    def reset(self):
        return self.env.reset()

    def step(self, action):
        action = self.map_action(action)
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            info = {'true_done': done}
            if done: break
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick : 
            zeros = np.zeros(self.image_shape)
            result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
            result = np.concatenate((result,np.array(self.history)))
        else: 
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick: self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))

class BreakOut:

    def __init__(self, crop = (34, -16, 8, -8), downscaling_factor = (2, 2), history_pick = 1, skip_frames = 1):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
        self.history_pick = history_pick
        self.state_space_size = history_pick*np.prod(self.image_shape)
        self.action_space_size = self.env.action_space.n
        self.state_shape = [None, self.history_pick] + list(self.image_shape)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0:2, 1:3, 2:0}
        self.crop = crop
        self.downscaling_factor = downscaling_factor

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def reset(self):
        self.env.reset()
        # First action should be to fire
        return self.step(1)[0]

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        info = {'true_done': done}
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick : 
            zeros = np.zeros(self.image_shape)
            result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
            result = np.concatenate((result,np.array(self.history)))
        else: 
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick: self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))

env_dict = {
    "Classic_Control": Classic_Control,
    "Pong": Pong,
    "CarRacing": CarRacing,
    "BreakOut": BreakOut
 }


# class Image_Based:
#     def __init__(self, game, crop, downscaling_factor, action_space_size, history_pick = 1, skip_frames = 1):
#         self.env = gym.make(game)
#         self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
#         self.history_pick = history_pick
#         self.state_space_size = history_pick*np.prod(self.image_shape)
#         self.action_space_size = action_space_size
#         self.state_shape = [None, self.history_pick] + list(self.image_shape)
#         self.history = []
#         self.skip_frames = skip_frames
#         self.action_dict = dict(zip(range(action_space_size), range(action_space_size)))
#         self.crop = crop
#         self.downscaling_factor = downscaling_factor

#     def set_action_dict(self, dict):
#         self.action_dict = dict

#     def map_action(self, action):
#         return self.action_dict[action]

#     def sample_action_space(self):
#         return random.sample(range(self.action_space_size), 1)[0]

#     def reset(self):
#         return self.env.reset()

#     def step(self, action):
#         action = self.map_action(action)
#         for i in range(self.skip_frames):
#             next_state, reward, done, info = self.env.step(action)
#             info = {'true_done': done}
#             if done: break
#         return next_state, reward, done, info

#     def render(self):
#         self.env.render()

#     def process(self, state):
#         self.add_history(state, None, None)
#         if len(self.history) < self.history_pick : 
#             zeros = np.zeros(self.image_shape)
#             result = np.tile(zeros,((self.history_pick - len(self.history)),1,1))
#             result = np.concatenate((result,np.array(self.history)))
#         else: 
#             result = np.array(self.history)
#         return result

#     def add_history(self, state, action, reward):
#         if len(self.history) >= self.history_pick: self.history.pop(0)
#         self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))
