import numpy as np
import gym
import math
import random
import matplotlib.pyplot as plt
import copy
import subprocess

env = gym.make('CartPole-v0')

import tensorflow as tf
sess = tf.Session()

# Parameters
# learning_rate = 0.001
# batch_size = 32

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
num_input = 4 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], stddev = 1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev = 1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes], stddev = 1))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev = 1)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev = 1)),
    'bOut': tf.Variable(tf.random_normal([num_classes], stddev = 1))
}

# concatenate the list of trainable variables
trainable_variables = weights.values() + biases.values()

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['bOut']
    return out_layer

# n x m matrix is equal to [[row_1], ... , [row_n]] where row_i is a m-dimensional vector


# X is a n.num_input = n.4 matrix
X = tf.placeholder("float")
# If X is n.4 then Q is n.2
Q = neural_net(X)
# Y is a n.1 column vector 
Y = tf.placeholder("float")
action = tf.placeholder("int32")
max_Q = tf.reduce_max(Q, axis = 1)
argmax_Q = tf.argmax(Q, axis = 1)

# print('Im here too ', tf.shape(Q[0][1]))

difference = tf.subtract(Y, tf.gather(Q, action, axis = 1))
loss = tf.reduce_sum(tf.square(difference))
# for weight in weights.values():
# 	loss = loss + 0.01*tf.reduce_sum(tf.square(weight))

# optimizer = tf.train.Grad(learning_rate = 0.1)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train_op = optimizer.minimize(loss)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.graph.finalize()

def phi(x): 
	return [x]

def eGreedy(state, epsilon):
	return sess.run(argmax_Q, feed_dict = {X: state})[0] if random.random() > epsilon else int(2*random.random())

def getEpsilon(time, nEpisodes):
	return 1 - time/nEpisodes

def greedy(state):
	return int(sess.run(argmax_Q, feed_dict = {X: state}))

def pause(): programPause = raw_input("Press the <ENTER> key to continue...")

def test(visualise = False, testEpisodes = 10):
	totalReward = 0
	for episode in range(testEpisodes):
		initObservation = env.reset()
		state = phi(initObservation)
		done = False
		
		while not done:
			if visualise: env.render()
			action = greedy(state)
			newObservation, reward, done, info = env.step(action)
			state = phi(newObservation)
			totalReward += reward
	return totalReward / float(testEpisodes)
	
nEpisodes = 3000
display_step = 20
discount = 0.99
epsilon = 1
replay_memory = []
replay_capacity = 1000
minibatch_size = 10

time = 0
for episode in range(nEpisodes):
	initObservation = env.reset()
	if len(replay_memory) > replay_capacity*2:
		replay_memory = replay_memory[replay_capacity:]
	
	# state is a 1.4 matrix, of the form [[row]]
	state = phi(initObservation)
	done = False
	fixed_weights = sess.run(trainable_variables)
	# print(fixed_weights)
	# pause()
	while not done:
		a = eGreedy(state, getEpsilon(time, nEpisodes))

		newObservation, reward, done, _ = env.step(a)
		newState = phi(newObservation)
		replay_memory.append((state, a, newState, reward))

		states = [None] * minibatch_size
		newStates = [None] * minibatch_size
		actions = [None] * minibatch_size
		rewards = [None] * minibatch_size
		i = 0
		for index in np.random.choice(a = len(replay_memory), size = minibatch_size):
			temp = replay_memory[index]
			states[i] = temp[0][0]
			newStates[i] = temp[2][0]
			actions[i] = temp[1]
			rewards[i] = temp[3]
			i += 1

		dict = {X: newStates}
		dict.update(zip(trainable_variables, fixed_weights))
		max_q = sess.run(max_Q, feed_dict = dict)
		y = np.matrix([rewards]).T + discount*np.matrix(max_q).T if not done else reward

		sess.run(train_op, feed_dict = {X: states, Y: y, action : actions})
		state = newState
		time += 1
	p = (episode + 1)%display_step == 0
	if p: 
		# print(episode)
		temp = test()
		print(temp)
		if temp > 50:
			test(visualise = True, testEpisodes = 5)

test(visualise = True)

print(sess.run(trainable_variables))

sess.close()