# MIT License

# Copyright (c) 2016 

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# github repo: https://github.com/takoika/PrioritizedExperienceReplay

import sys
sys.dont_write_bytecode = True

import numpy as np
import random
import sumtree
from utils import pause


class Prioritized_Replay_Memory():

    def __init__(self, memory_capacity, batch_size, alpha=0.6):
        self.is_full = False
        self.tree = sumtree.SumTree(memory_capacity)
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.alpha = alpha

    def length(self):
        return self.tree.filled_size()

    def full(self):
        if self.is_full: return True
        self.is_full = self.tree.filled_size() == self.memory_capacity
        return self.is_full

    def get_q_grid(self, size, training_metadata):
        grid = self.get_mini_batch(training_metadata)[0]
        return grid

    def add(self, agent, state, action, reward, next_state, done):
        one_hot_action = np.zeros(agent.action_size)
        one_hot_action[action] = 1
        y = reward + (agent.discount * agent.sess.run(agent.Q_value_at_action, feed_dict={agent.state_tf: [next_state], agent.action_tf: [one_hot_action]}))
        td_error = agent.sess.run(agent.td_error, feed_dict={agent.y_tf: y, agent.state_tf: [state], agent.action_tf: [one_hot_action]})
        data = (state, one_hot_action, reward, next_state, done)
        self.tree.add(data, td_error**self.alpha)

    def get_mini_batch(self, training_metadata, beta = 0.8):
        out = []
        indices = []
        weights = []
        priorities = []
        segment = self.tree.total() / float(self.batch_size)
        for i in range(self.batch_size):
            r = i + random.random()*segment
            data, priority, index = self.tree.get(r)
            priorities.append(priority)
            weights.append((1. / self.memory_capacity / priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
        weights /= max(weights)  # Normalize for stability

        state_batch = [d[0] for d in out]
        action_batch = [d[1] for d in out]
        reward_batch = [d[2] for d in out]
        next_state_batch = [d[3] for d in out]
        done_batch = [d[4] for d in out]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, map(int, indices)

    def priority_update(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.tree.update(idx, priority**self.alpha)

    def reset_alpha(self, alpha):
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)
