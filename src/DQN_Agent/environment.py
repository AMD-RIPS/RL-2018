import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import random


class Environment:

    def __init__(self, game):
        self.env = gym.make(game)
        self.state_space_size = np.shape(self.env.observation_space)[0]
        self.state_space_lower_bounds = self.env.observation_space.low
        self.state_space_upper_bounds = self.env.observation_space.high
        self.action_space_size = self.env.action_space.n
        self.history = []

    def sample_action_space(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def process(self, x):
        # Note that in processing we might want to use self.history - e.g. for last 4 frames x only contains most recent frames, the other 3 we must get from self.history
        return x

    def add_history(self, state, action, reward):
        pass

