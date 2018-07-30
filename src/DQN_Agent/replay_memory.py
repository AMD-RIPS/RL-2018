import sys
sys.dont_write_bytecode = True

import numpy as np
import random
from utils import pause
import sumtree


class Replay_Memory:

    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []

    def length(self):
        return len(self.memory)

    def get_mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]
        done_batch = [data[4] for data in mini_batch]
        weights = indices = [1 for _ in mini_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices

    def add(self, agent, state, action, reward, next_state, done):
        one_hot_action = np.zeros(agent.action_size)
        one_hot_action[action] = 1
        self.memory.append((state, one_hot_action, reward, next_state, done))
        if (len(self.memory) > self.memory_capacity):
            self.memory.pop(0)

    def get_q_grid(self, size):
        return [data[0] for data in random.sample(self.memory, size)]

    def priority_update(self, *args):
        pass
