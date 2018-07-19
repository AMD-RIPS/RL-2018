import sys
sys.dont_write_bytecode = True

import numpy as np
import random

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


class Replay_Memory:
	def __init__(self, memory_capacity, batch_size, q_grid_size):
		self.memory_capacity = memory_capacity
		self.batch_size = batch_size
		self.memory = []
		self.history = []
		self.q_grid_size = q_grid_size

	def length(self):
		return len(self.memory)

	def get_batch(self):
		mini_batch = random.sample(self.memory, self.batch_size)
		state_batch = [data[0] for data in mini_batch]
		action_batch = [data[1] for data in mini_batch]
		reward_batch = [data[2] for data in mini_batch]
		next_state_batch = [data[3] for data in mini_batch]
		done_batch = [data[4] for data in mini_batch]
		return state_batch, action_batch, reward_batch, next_state_batch, done_batch

	def add(self, environment, state, action, reward, next_state, done, action_size):
		one_hot_action = np.zeros(action_size) 
		one_hot_action[action] = 1
		processed_state = environment.process(state)
		processed_next_state = environment.process(next_state)
		self.memory.append((processed_state, one_hot_action, reward, processed_next_state, done))
		if (len(self.memory) > self.memory_capacity):
			self.memory.pop(0)

	def get_q_grid(self):
		return [data[0] for data in random.sample(self.memory, self.q_grid_size)]