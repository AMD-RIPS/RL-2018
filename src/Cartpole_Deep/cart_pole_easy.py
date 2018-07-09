import gym
import numpy as np
import tensorflow as tf
# from tf_arch import *

# Store layers weight & bias
num_input = 4 # Number of cartpole states
n_hidden_1 = 10
n_hidden_2 = 10
num_classes = 2
weights = {
    'w1': tf.Variable(tf.random_normal(shape=[num_input, n_hidden_1],dtype=tf.float32)),
    # 'w2': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2],dtype=tf.float32)),
    'out': tf.Variable(tf.random_normal(shape=[n_hidden_2, num_classes],dtype=tf.float32))
}


def Q_neural_net(x):
    x = tf.constant(x, shape=[1,4], dtype='float32')
    # print sess.run(x)
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.matmul(x, weights['w1'])
    # Hidden fully connected layer with 256 neurons
    # layer_2 = tf.matmul(layer_1, weights['w2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) 
    return out_layer

def get_opt_action(state):
	return sess.run(tf.argmax(Q_neural_net(state),axis=1))[0]

# returns action based on epsilon greedy policy
def get_action(state, epsilon, num_actions):
	if (np.random.uniform() < epsilon):
		return np.random.randint(0, num_actions)
	else:
		return get_opt_action(state)

# return random minibatch of transitions from D
# def minibatch(D):
# 	n = np.shape(D)[0]
# 	b_size = np.min(n,3)
# 	sample_indices = np.random.choice(n, size=b_size, replace=False)
# 	return D[sample_indices]

# return max Q for action
def maxQ(state):
	return tf.reduce_max(Q_neural_net(state), axis=None)


# PWNW 
def gradient(state, action, y_j):
	action = tf.constant(action)
	# temp = Q_neural_net(state)
	# print 'Q values: ', sess.run(temp)
	q_val = tf.gather(tf.transpose(tf.gather(Q_neural_net(state), indices=0)), indices=action)
	# print sess.run(q_val)
	loss = tf.square(tf.subtract(y_j, q_val))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	grad = optimizer.minimize(loss)
	# print '***', sess.run(tf.shape(grad[0]))
	return grad



env = gym.make('CartPole-v0')
num_actions = env.action_space.n

N = 5		# replay memory capacity
D = []			# replay memory
num_episodes = 100
epsilon = .3
discount = 0.99
learning_rate = .05

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for episodes in range(num_episodes):
	state = env.reset()
	done = False
	print 'Episode: {}'.format(episodes)
	tot_reward = 0
	while not done:
		action = get_action(state, epsilon, num_actions)
		# print 'Action: {}'.format(action)
		new_state, reward, done, _ = env.step(action)
		if done:
			y = tf.constant(reward)
		else:
			y = reward + discount*maxQ(new_state)
		gradient(state, action, y)
		state = new_state
		if (episodes > (num_episodes - 10)):
			env.render()
		tot_reward += reward
	print tot_reward
		# print current_gradient
	
		# compute average of gradients
		# update weights of Q	
sess.close()
