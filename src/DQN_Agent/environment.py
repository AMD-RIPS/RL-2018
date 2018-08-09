import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time


class Classic_Control:

    def __init__(self, game='CartPole-v1'):
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

    def process(self, x):
        return x

    def add_history(self, state, action, reward):
        pass

    def __str__(self):
        return self.name


class Pong:

    def __init__(self, crop=(34, -16, 8, -8), downscaling_dimension=(84, 84), history_pick=4, skip_frames=4):
        game_version = 'PongNoFrameskip-v4'
        self.name = game_version + '_' + str(time.time())
        self.env = gym.make(game_version)
        self.env = gym.wrappers.Monitor(self.env, '../../videos/pong_deterministic')
        self.downscaling_dimension = downscaling_dimension
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.downscaling_dimension)
        self.action_space_size = 3
        self.state_shape = [None, self.history_pick] + list(self.downscaling_dimension)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0: 0, 1: 3, 2: 2}
        self.crop = crop

    def map_action(self, action):
        return self.action_dict[action]

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def reset(self):
        return self.process(self.env.reset())

    def step(self, action):
        action = self.map_action(action)
        total_reward = 0
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            info = {'true_done': done}
            total_reward += reward
            if reward == -1: done = True
            if done:
                break
        return self.process(next_state), total_reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.downscaling_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_dimension))

    def __str__(self):
        return self.name


class CarRacing:

    def __init__(self, type="CarRacing", crop=(None, None, None, None), downscaling_dimension=(84, 84), history_pick=4, seed=None, test=False):
        self.name = type + str(time.time())
        self.env = gym.make(type + '-v0')
        self.downscaling_dimension = downscaling_dimension
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.downscaling_dimension)
        self.action_space_size = 5
        self.state_shape = [None, self.history_pick] + list(self.downscaling_dimension)
        self.history = []
        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
        self.crop = crop
        self.seed = seed
        self.test = test

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_dict[action]

    def reset(self):
        if self.seed:
            self.env.seed(random.choice(self.seed))
        return self.process(self.env.reset())

    def step(self, action):
        action = self.map_action(action)
        total_reward = 0
        n = 1 if self.test else random.choice([2, 3, 4])
        for i in range(n):
            next_state, reward, done, info = self.env.step(action)
            if not self.test:
                reward = self.clip_reward(reward)
            total_reward += reward
            info = {'true_done': done}
            if done: break
        return self.process(next_state), total_reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.downscaling_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        self.history.append(utils.process_image(state, self.crop, self.downscaling_dimension))

    def clip_reward(self, reward):
        if reward > 0:
            clipped_reward = reward/20
        else:
            clipped_reward = -1
        return clipped_reward

    def __str__(self):
    	return self.name + '\nseed: {0}\nactions: {1}\n'.format(self.seed, self.action_dict)

class BreakOut:

    def __init__(self, crop=(34, -16, 8, -8), downscaling_dimension = (84, 84), history_pick=4, skip_frames=4):
        game_version = 'BreakoutNoFrameskip-v4'
        self.name = game_version + '_' + str(time.time())
        self.env = gym.make(game_version)
        self.downscaling_dimension = downscaling_dimension
        self.history_pick = history_pick
        self.state_space_size = history_pick * np.prod(self.downscaling_dimension)
        self.action_space_size = self.env.action_space.n
        self.state_shape = [None, self.history_pick] + list(self.downscaling_dimension)
        self.history = []
        self.skip_frames = skip_frames
        self.action_dict = {0: 0, 1: 1, 2: 2, 3: 3}
        self.crop = crop

    def map_action(self, action):
        return self.action_dict[action]

    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def reset(self):
        self.env.reset()
        self.life_remaining = 5
        # First action should be to fire
        return self.step(1)[0]

    def step(self, action):
        action = self.map_action(action)
        total_reward = 0
        for i in range(self.skip_frames):
            next_state, reward, done, info = self.env.step(action)
            total_reward += reward
            info.update({'true_done': done})
            if info['ale.lives'] < self.life_remaining:
                done = True
            if info['ale.lives'] == 0:  
                break
        self.life_remaining = info['ale.lives']
        return self.process(next_state), total_reward, done, info

    def render(self):
        self.env.render()

    def process(self, state):
        self.add_history(state, None, None)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.downscaling_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state, action, reward):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        self.history.append(utils.process_nature_atari(state))

    def __str__(self):
        return self.name


env_dict = {
    "CartPole": Classic_Control,
    "Pong": Pong,
    "CarRacing": CarRacing,
    "BreakOut": BreakOut
}
