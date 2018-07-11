import gym
import numpy as np
import tensorflow as tf
import random

class Playground:
	def __init__(self, game, num_hidden_layers, layer_sizes, epsilon_max, epsilon_min, alpha, gamma, batch_size):
		self.game = game
		self.num_hidden_layers = num_hidden_layers
		self.layer_sizes = layer_sizes
		self.epsilon_max = epsilon_max
		self.epsilon_min = epsilon_min
		self.alpha = alpha
		self.gamma = gamma
		self.batch_size = batch_size

	def Q_nn(self, input):
		with tf.device('/device:CPU:0'):
			neural_net = input
			for n in self.layer_sizes:
				neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
			return tf.layers.dense(neural_net, self.action_size, activation=None)

	# Initialize tensorflow place holders
	def initialize_tf_variables(self):
		self.state_tf = tf.placeholder(shape=[None, self.state_size],dtype=tf.float64)
		self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float64)
		self.y_tf = tf.placeholder(dtype=tf.float64)
		self.Q_value = self.Q_nn(self.state_tf)
		self.Q_argmax = tf.argmax(self.Q_value[0])
		self.Q_amax = tf.reduce_max(self.Q_value[0])
		self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf),reduction_indices = 1)
		self.loss = tf.reduce_mean(tf.square(self.y_tf - self.Q_value_at_action))
		self.train_op = tf.train.AdamOptimizer(learning_rate = self.alpha).minimize(self.loss)
		config = tf.ConfigProto()
		config.allow_soft_placement=True
		config.gpu_options.allow_growth = True
		config.log_device_placement = True
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())

	def experience_replay(self):
		mini_batch = random.sample(self.history, self.batch_size)
		state_batch = [data[0] for data in mini_batch]
		action_batch = [data[1] for data in mini_batch]
		reward_batch = [data[2] for data in mini_batch]
		next_state_batch = [data[3] for data in mini_batch]
		y_batch = []
		Q_value_batch = self.sess.run(self.Q_value, feed_dict={self.state_tf: next_state_batch})
		for i in range(self.batch_size):
			done = mini_batch[i][4]
			if done:
				y_batch.append(reward_batch[i])
			else :
				y_batch.append(reward_batch[i] + self.gamma*np.max(Q_value_batch[i]))

		self.sess.run(self.train_op, feed_dict={self.y_tf: y_batch, self.action_tf: action_batch, self.state_tf: state_batch})


	def get_action(self, state, epsilon):
		if random.random() < epsilon:
				return self.env.action_space.sample()
		else:
			return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})

	def get_epsilon(self, episode):
		eps_decay_rate = (self.epsilon_min-self.epsilon_max)/(self.num_episodes/2)
		eps = max(self.epsilon_max + eps_decay_rate*episode, self.epsilon_min)
		return eps

	def rand_state_sample(self):
		sample = np.zeros(self.state_size)
		for i in range(self.state_size):
			sample[i] = np.random.uniform(self.lower_bounds[i], self.upper_bounds[i])
		return [sample]

	def estimate_avg_q(self, num_samples):
		q_avg = 0.0
		for i in range(num_samples):
			state_sample = self.rand_state_sample()
			q_avg += np.mean(self.sess.run(self.Q_value, feed_dict={self.state_tf: state_sample}))
		q_avg /= num_samples
		return q_avg

	def begin_training(self, num_episodes):
		self.env = gym.make(self.game)
		self.state_size = np.shape(self.env.observation_space)[0]
		self.lower_bounds = self.env.observation_space.low
		self.upper_bounds = self.env.observation_space.high
		self.action_size = self.env.action_space.n
		self.initialize_tf_variables()
		self.num_episodes = num_episodes
		q_averages = np.zeros(num_episodes)
		self.history = []
		max_history = 10000
		for episode in range(num_episodes):
			done = False
			tot_reward = 0
			state = self.env.reset()
			if (len(self.history) > max_history):
				self.history.pop(0)
			while not done:
				action = self.get_action(state, self.get_epsilon(episode))
				next_state, reward, done, _ = self.env.step(action)
				one_hot_action = np.zeros(self.action_size)
				one_hot_action[action] = 1
				self.history.append((state, one_hot_action, reward, next_state, done))
				if done:
					y = reward
				else:
					y = reward + self.gamma*self.sess.run(self.Q_amax, feed_dict={self.state_tf: [next_state]})
				tot_reward += reward
				if len(self.history) < self.batch_size:
					continue
					# self.sess.run(self.train_op, feed_dict={self.state_tf:[state], self.y_tf: y, self.action_tf: [one_hot_action]})
				else:
					self.experience_replay()
				state = next_state
			q_averages[episode] = self.estimate_avg_q(1000)
			print 'Episode: {}. Reward: {}. Epsilon: {}'.format(episode,tot_reward, self.get_epsilon(episode))
		file_name = 'avg_q_' + self.game + '.csv'
		np.savetxt(file_name, q_averages, delimiter=',')