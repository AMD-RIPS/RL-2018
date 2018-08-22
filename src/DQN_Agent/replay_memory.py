####################################################################
# This file defines the Replay_Memory class. There are five        #
# functions: __init__(memory_capacity, batch_size) to set up the   #
# replay memory with proper storage size and size of the minibatch,#
# length() to output the size of filled memory, get_mini_batch to  #
# randomly sample a batch with the size defined in batch_size,     #
# add(agent, state, action, reward, next_state, done) to add a new #
# episode into the memory and get_q_grid(size) to get the sample   #
# for evaluating average Q value during training                   #
####################################################################

# import necesary files and functions
import sys
sys.dont_write_bytecode = True

import numpy as np
import random
from utils import pause

class Replay_Memory:

    # To initialize the replay memory
    # memory_capacity defines the limit for storage
    # batch_size defines the size for the minibatch
    def __init__(self, memory_capacity, batch_size):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.memory = []

    # To output the filled size of memory
    def length(self):
        return len(self.memory)
        
    # To randomly sample a minibatch for update
    def get_mini_batch(self, *args, **kwargs):
        mini_batch = random.sample(self.memory, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]
        done_batch = [data[4] for data in mini_batch]
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    # To add an episode into the memory
    # agent defines the environment
    # state defines the past stack of images
    # action defines the action taken
    # reward defines the reward returned based on the state and action
    # next_state defines the current stack of images
    def add(self, agent, state, action, reward, next_state, done):
        # convert the action to one hot action for easier computation
        one_hot_action = np.zeros(agent.action_size)
        one_hot_action[action] = 1
        self.memory.append((state, one_hot_action, reward, next_state, done))
        # delete the earliest episode if memory is full
        if (len(self.memory) > self.memory_capacity):
            self.memory.pop(0)

    # To randomly sample states for evaluating average Q value during 
    # training.
    # size defines the size to sample.
    def get_q_grid(self, size, *args, **kwargs):
        return [data[0] for data in random.sample(self.memory, size)]