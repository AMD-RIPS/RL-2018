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
learning_rate = 0.001
# batch_size = 32
display_step = 20

# Network Parameters
n_hidden_1 = 10 # 1st layer number of neurons
n_hidden_2 = 10 # 2nd layer number of neurons
num_input = 4 # MNIST data input (img shape: 28*28)
num_classes = 2 # MNIST total classes (0-9 digits)

X = tf.placeholder("float")
Y = tf.placeholder("float")
action = tf.placeholder("int32")

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1], stddev = 0.01)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev = 0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes], stddev = 0.01))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev = 0.01)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2], stddev = 0.01)),
    'bOut': tf.Variable(tf.random_normal([num_classes], stddev = 0.01))
}

trainable_variables = weights.values() + biases.values()

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    out_layer = tf.matmul(layer_2, weights['out']) + biases['bOut']
    return out_layer

Q = neural_net(X)



# print('Im here too ', tf.shape(Q[0][1]))
difference = tf.subtract(Y, tf.gather(Q, action, axis = 1))
loss = tf.reduce_sum(tf.square(difference))
for weight in weights.values():
	loss = loss + 0.01*tf.reduce_sum(tf.square(weight))

optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.1)
# optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# d = {X: [[1.3, 2.0, 3.2, 4.2]], Y : [1], action : 0}

# print(sess.run(Q, feed_dict = d))
# print(sess.run(tf.gather(Q, action), feed_dict = d))
# print(sess.run(difference, feed_dict = d))


# # print(sess.run(loss_1, feed_dict = {Y: [9], X: [[1.3, 2.0, 3.2, 4.2]]}))

def phi(x): 
	return [x]

def eGreedy(state, epsilon):
	q = sess.run(Q, feed_dict = {X: state})
	# print('q = {0}, greedy action = {1}'.format(q, np.argmax(q)))
	return np.argmax(q) if random.random() > epsilon else int(2*random.random())

def getEpsilon():
	return 0.3

def greedy(state):
	return np.argmax(sess.run(Q, feed_dict = {X: state}))

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
	return totalReward / 10.0
	
discount = 0.99
epsilon = 0.5
nEpisodes = 200
replay_memory = []
# states = []
replay_capacity = 1000
minibatch_size = 32

fixed_weights = sess.run(trainable_variables)
for episode in range(nEpisodes):
	initObservation = env.reset()
	state = phi(initObservation)
	done = False
	
	while not done:
		a = eGreedy(state, getEpsilon())

		newObservation, reward, done, _ = env.step(a)
		newState = phi(newObservation)
		replay_memory.append((state, action, newState, reward))

		for index in np.random.choice(a = len(replay_memory), size = minibatch_size):
			reward = replay_memory[index][3]
			nState = replay_memory[index][2]
			dict = {X: nState}
			dict.update(zip(trainable_variables, fixed_weights))
			q_value = sess.run(Q, feed_dict = dict)

			y = reward + discount * np.amax(q_value) if not done else reward
			sess.run(train_op, feed_dict = {X: state, Y: [y], action : a})
		state = newState
	fixed_weights = sess.run(trainable_variables)
	p = episode%display_step == 0
	if p: 
		temp = test()
		print(temp)
		if temp > 50:
			test(visualise = True, testEpisodes = 5)

test(visualise = True)

print(sess.run(trainable_variables))

sess.close()