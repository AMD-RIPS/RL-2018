import gym
import numpy as np
import tensorflow as tf

# Computes phi function as defined in paper.
# Returns last 4 elements in history array
# Parameters:
# - history: A vector of tuples where each element is a
#            (state, action) pair 
# 			Example: [[[1,2,3,4],0], [[1,1,1,1],1]]
def phi(history):
	hist_size = np.shape(history)[0]
	return_size = np.amin([4, hist_size])
	ret_vec = history[(hist_size - return_size):]
	for i in range(return_size):
		ret_vec[i] = ret_vec[i][0]
	return ret_vec

# returns action based on epsilon greedy policy
def get_action(phi_s, epsilon, num_actions):
	if (np.random.uniform() < epsilon):
		return np.random.randint(0, num_actions)
	else:
		# get greedy action

# return random minibatch of transitions from D
def minibatch(D):
	return D

# return max Q for action
def maxQ(phi_j):
	return 0

def gradient(phi_j, action, y_j):
	pass


env = gym.make('CartPole-v0')
num_actions = env.action_space.n

N = 1000		# replay memory capacity
D = []			# replay memory
num_episodes = 100
epsilon = 0.1
discount = 0.99

for episodes in range(num_episodes):
	x1 = env.reset()
	history = [[x1,0]]
	done = False

	while not done:
		phi_t = phi(history)
		action = get_action(phi_t, epsilon, num_actions)
		x_t, r_t, done, _ = env.step(action)
		history.append([x_t,action])
		D.append([phi_t, action, r_t, phi(history), done])
		
		batch = minibatch(D)
		batch_size = np.shape(batch)[0]
		y = np.zeros(batch_size)
		gradients = np.zeros(batch_size)

		for j in range(batch_size):
			if batch[j][4]:			# terminal
				y[j] = batch[j][2]
			else:
				y[j] = batch[j][2] + discount * maxQ(batch[j][3])

			gradients[j] = gradient(batch[j][0], batch[j][1], y[j])
	
		# compute average of gradients
		# update weights of Q	


for i in range(1000):
	env.render()
	env.step(env.action_space.sample())
