import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random
from PIL import Image
import utils
import time

class CarRacing:

    # Parameters
    # - seed: List of seeds to sample from during training. Default is none (random games)
    def __init__(self, seed=None):
        self.name = "CarRacing" + str(time.time())
        self.env = gym.make("CarRacing-v0")
        self.image_dimension = [84,84]
        self.history_pick = 4
        self.state_space_size = self.history_pick * np.prod(self.image_dimension)
        self.state_shape = [None, self.history_pick] + list(self.image_dimension)
        self.action_dict = {0: [-1, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 0.8], 4: [0, 0, 0]}
        self.action_space_size = len(self.action_dict)
        self.history = []
        self.seed = seed
        self.curve_data = {}

    # returns a random action
    def sample_action_space(self):
        return np.random.randint(self.action_space_size)

    def map_action(self, action):
        return self.action_dict[action]

    # resets the environment and returns the initial state
    def reset(self, test=False):
        if self.seed and not test:
            self.env.seed(random.choice(self.seed))
        return self.process(self.env.reset())

    # take action 
    def step(self, action, test=False):
        action = self.map_action(action)
        total_reward = 0
        n = 1 if test else random.choice([2, 3, 4])
        for i in range(n):
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done: break
        return self.process(next_state), total_reward, done, None

    def test_curves(self):
        curves = self.env.env.test_curves()
        for curve in curves:
            curve_type = curve[2] + str(curve[3])
            if curve_type in self.curve_data:
                self.curve_data[curve_type][0] += curve[0]
                self.curve_data[curve_type][1] += curve[1]
            else:
                self.curve_data[curve_type] = curve[:2]
        return self.curve_data

    def analyze_curves(self, path):
        f = open(path, 'w')
        for direction in ['L', 'R', 'S']:
            for i in range(1, 6):
                curve = direction + str(i)
                if curve in self.curve_data:
                    visited = self.curve_data[curve][0]
                    total = self.curve_data[curve][1]
                    pct = visited / float(total)
                    f.write(','.join([curve, str(visited), str(total), str(pct)]) + '\n')
                else:
                    f.write(','.join([curve, 'None', 'None', 'None']) + '\n')
        f.close()
        self.curve_data = {}

    def render(self):
        self.env.render()

    # process state and return the current history
    def process(self, state):
        self.add_history(state)
        if len(self.history) < self.history_pick:
            zeros = np.zeros(self.image_dimension)
            result = np.tile(zeros, ((self.history_pick - len(self.history)), 1, 1))
            result = np.concatenate((result, np.array(self.history)))
        else:
            result = np.array(self.history)
        return result

    def add_history(self, state):
        if len(self.history) >= self.history_pick:
            self.history.pop(0)
        temp = utils.process_image(state)
        self.history.append(temp)

    def __str__(self):
    	return self.name + '\nseed: {0}\nactions: {1}'.format(self.seed, self.action_dict)