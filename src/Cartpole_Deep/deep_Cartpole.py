import numpy as np
import gym
import math
import random
import matplotlib.pyplot as plt
import copy
import subprocess

env = gym.make('CartPole-v0')

import tensorflow as tf

# Network Parameters
n_hidden_1 = 64 # 1st layer number of neurons
n_hidden_2 = 64 # 2nd layer number of neurons
num_input = 4 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)

# An n x m matrix is of the form [[row_1], ... , [row_n]] where row_i is a m-dimensional vector
# X is a n.num_input = n.4 matrix
X = tf.placeholder("float", [None, num_input])
# If X is n.4 then Q is n.2

# Definition of the neural net
layer1 = tf.layers.dense(inputs = X, units = n_hidden_1, activation = tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer)
layer2 = tf.layers.dense(inputs = layer1, units = n_hidden_2, activation = tf.nn.relu, kernel_initializer = tf.truncated_normal_initializer)
Q = tf.layers.dense(inputs = layer2, units =  num_classes)

trainable_variables = tf.trainable_variables()

# Y is a n.1 column vector 
Y = tf.placeholder("float")
action = tf.placeholder("int32")
learning_rate = tf.placeholder("float")
max_Q = tf.reduce_max(Q, axis = 1)
argmax_Q = tf.argmax(Q, axis = 1)

Q_actionvals = tf.matmul(Q, tf.transpose(tf.one_hot(indices = action, depth = num_classes)))
loss = tf.losses.mean_squared_error(Y, Q_actionvals)

# optimizer = tf.train.RMSPropOptimizer(learning_rate)
# optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate)
train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
sess.graph.finalize()

def phi(x): 
	return [x]

def eGreedy(state, epsilon):
	return sess.run(argmax_Q, feed_dict = {X: state})[0] if random.random() > epsilon else int(2*random.random())

def getEpsilon(episode, nEpisodes):
	return 0.5 - 0.5*episode/nEpisodes

def getLearningRate(time, nEpisodes):
	return 0.001

def greedy(state):
	return sess.run(argmax_Q, feed_dict = {X: state})[0]

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
	
nEpisodes = 2000
display_step = 20
discount = 0.99
replay_memory = []
replay_capacity = 1000
minibatch_size = 16

time = 0
for episode in range(nEpisodes):
	initObservation = env.reset()
	if len(replay_memory) > replay_capacity*2:
		replay_memory = replay_memory[replay_capacity:]
	
	# state is a 1.4 matrix, of the form [[row]]
	state = phi(initObservation)
	done = False
	fixed_weights = sess.run(trainable_variables)
	while not done:
		a = eGreedy(state, getEpsilon(episode, nEpisodes))

		newObservation, reward, done, _ = env.step(a)
		newState = phi(newObservation)
		replay_memory.append((state, a, newState, reward, done))
		# filling up replay memory
		if episode < 2: break

		states = [None] * minibatch_size
		newStates = [None] * minibatch_size
		actions = [None] * minibatch_size
		rewards = [None] * minibatch_size
		dones = [None] * minibatch_size
		i = 0
		for index in np.random.choice(a = len(replay_memory), size = minibatch_size):
			temp = replay_memory[index]
			states[i] = temp[0][0]
			newStates[i] = temp[2][0]
			actions[i] = temp[1]
			rewards[i] = temp[3]
			dones[i] = temp[4]
			i += 1

		dict = {X: newStates}
		dict.update(zip(trainable_variables, fixed_weights))
		max_q = sess.run(max_Q, feed_dict = dict)
		# y = np.matrix([rewards]) + discount*np.matrix(max_q) if not done else np.matrix([rewards])
		y = np.matrix([[rewards[i] + discount*max_q[i] if not dones[i] else rewards[i] for i in range(minibatch_size)]])
		y = y.T
		train_dict = {X: states, Y: y, action : actions, learning_rate: 0.01}
		sess.run(train_op, feed_dict = train_dict)
		state = newState
		time += 1
	score = test(testEpisodes = 10)
	if score == 200: break
	print("score = {0}, epsilon = {1}, learning rate = {2}".format(score, getEpsilon(episode, nEpisodes), getLearningRate(time, nEpisodes)))

test(visualise = True)

print(sess.run(trainable_variables))

sess.close()