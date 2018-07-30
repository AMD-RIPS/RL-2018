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
        self.history = []

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


#####################################################################################################

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

# Link to repo: https://github.com/takoika/PrioritizedExperienceReplay

class Prioritized_Replay_Memory():
    """ The class represents prioritized experience replay buffer.

    The class has functions: store samples, pick samples with 
    probability in proportion to sample's priority, update 
    each sample's priority, reset alpha.

    see https://arxiv.org/pdf/1511.05952.pdf .

    """
    
    def __init__(self, memory_capacity, batch_size, alpha = 1):
        """ Prioritized experience replay buffer initialization.
        
        Parameters
        ----------
        memory_capacity : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = sumtree.SumTree(memory_capacity)
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.alpha = alpha

    def length(self):
        return self.tree.filled_size()

    def get_q_grid(self, size):
        grid = self.get_mini_batch(beta=0)[0]
        return grid

    def add(self, agent, state, action, reward, next_state, done):
        """ Add new sample.
        
        Parameters
        ----------
        data : object
            new sample
        priority : float
            sample's priority
        """
        one_hot_action = np.zeros(agent.action_size)
        one_hot_action[action] = 1
        y = reward + (agent.discount*agent.sess.run(agent.Q_value_at_action, feed_dict={agent.state_tf: [next_state], agent.action_tf:[one_hot_action]}))
        td_error = agent.sess.run(agent.td_error, feed_dict={agent.y_tf: y, agent.state_tf: [state], agent.action_tf: [one_hot_action]})
        data = (state, one_hot_action, reward, next_state, done)
        self.tree.add(data, td_error**self.alpha)

    def get_mini_batch(self, beta = 0.8):
        """ The method return samples randomly.
        
        Parameters
        ----------
        beta : float
        
        Returns
        -------
        out : 
            list of samples
        weights: 
            list of weights
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        out = []
        indices = []
        weights = []
        priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            priorities.append(priority)
            weights.append((1./self.memory_capacity/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating
            
        
        self.priority_update(indices, priorities) # Revert priorities

        weights /= max(weights) # Normalize for stability
        state_batch = [data[0] for data in out]
        action_batch = [data[1] for data in out]
        reward_batch = [data[2] for data in out]
        next_state_batch = [data[3] for data in out]
        done_batch = [data[4] for data in out]
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, map(int, indices)

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.
        
        Parameters
        ----------
        indices : 
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)
    
    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.

        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [self.tree.get_val(i)**-old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)