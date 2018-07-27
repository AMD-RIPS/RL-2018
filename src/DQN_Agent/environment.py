import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time


class Classic_Control:

    def __init__(self, game):
        self.name = game + str(time.time())
        self.env = gym.make(game)
        self.state_space_size = np.shape(self.env.observation_space)[0]
        self.action_space_size = self.env.action_space.n
        self.history = []
        self.state_shape = [None, self.state_space_size]
        self.skip_frames = 1

    def sample_action_space(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.process(self.env.reset())

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        info = {'true_done': done}
        return self.process(next_state), reward, done, info

    def render(self):
        self.env.render()

    # Returns a (sample_space_size, ) vector
    def process(self, x):
        return x

    def add_history(self, state, action, reward):
        pass

    def __str__(self):
        return self.name


class Pong:

    def __init__(self, crop=(34, -16, 8, -8), downscaling_factor=(2, 2), history_pick=4, skip_frames=1):
        self.name = "Pong_" + str(time.time())
        self.env = gym.make('Pong-v0')
        self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.image_shape)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.image_shape)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0: 0, 1: 3, 2: 2}
        self.crop = crop
        self.downscaling_factor = downscaling_factor

    def map_action(self, action):
        return self.action_dict[action]

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def reset(self):
        return self.process(self.env.reset())

    def step(self, action):
        action = self.map_action(action)
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            info = {'true_done': done}
            if reward == -1: done = True
            if done:
                break
        return self.process(next_state), reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.image_shape)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))

    def __str__(self):
        return "Pong_" + str(time.time())


class CarRacing:

    def __init__(self, crop=(None, None, None, None), downscaling_factor=(2, 2), history_pick=4, skip_frames=4):
        self.name = "CarRacing" + str(time.time())
        self.env = gym.make('CarRacing-v0')
        self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.image_shape)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.image_shape)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8]}
        self.crop = crop
        self.downscaling_factor = downscaling_factor

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_dict[action]

    def reset(self):
        return self.process(self.env.reset())

    def step(self, action):
        action = self.map_action(action)
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            info = {'true_done': done}
            if done:
                break
        return self.process(next_state), reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.image_shape)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))


class BreakOut:

    def __init__(self, crop=(34, -16, 8, -8), downscaling_factor=(2, 2), history_pick=4, skip_frames=1):
        self.name = "BreakOut" + str(time.time())
        self.env = gym.make('BreakoutDeterministic-v4')
        self.image_shape = utils.get_image_shape(self.env, crop, downscaling_factor)
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.image_shape)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.image_shape)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0: 0, 2: 2, 1: 3}
        self.crop = crop
        self.downscaling_factor = downscaling_factor

    def map_action(self, action):
        return self.action_dict[action]

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def reset(self):
        self.env.reset()
        # First action should be to fire
        return self.step(1)[0]

    def step(self, action):
        paction = self.map_action(action)
        next_state, reward, done, _ = self.env.step(action)
        info = {'true_done': done}
        return self.process(next_state), reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.image_shape)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_factor))

env_dict = {
    "Classic_Control": Classic_Control,
    "Pong": Pong,
    "CarRacing": CarRacing,
    "BreakOut": BreakOut
}
