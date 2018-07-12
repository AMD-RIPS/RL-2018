import gym
import numpy as np
import tensorflow as tf
import random
from skimage.transform import downscale_local_mean

class Playground:

	def __init__(self, game, num_hidden_layers, layer_sizes, epsilon_max, epsilon_min, alpha, gamma, batch_size, memory_capacity,
		steering, acceleration, deceleration):
		self.game = game
		self.num_hidden_layers = num_hidden_layers
		self.layer_sizes = layer_sizes
		self.epsilon_max = epsilon_max
		self.epsilon_min = epsilon_min
		self.alpha = alpha
		self.gamma = gamma
		self.batch_size = batch_size
		self.memory_capacity = memory_capacity
		self.history_pick = 4
		self.steering = steering
		self.acceleration = acceleration
		self.deceleration = deceleration
		self.steering_size = len(self.steering)
		self.acceleration_size = len(self.acceleration)
		self.deceleration_size = len(self.deceleration)
		self.initialize_tf_variables()

	def rgb2gray(self, rgb):
		return np.dot(rgb, [0.299, 0.587, 0.114])
	
	def down_sample(self, state):
		state = self.rgb2gray(state)
		return  state#downscale_local_mean(state, (2, 2))

	def get_state_space_size(self, state):
		return np.shape(self.down_sample(state))

	def Q_nn(self, input):
		with tf.device('/device:CPU:0'):
			sess = tf.InteractiveSession()
			print tf.shape(input).eval()
			layer1_out = tf.layers.conv2d(input, filters = 16, kernel_size = [8, 8], strides = [4, 4], padding = 'SAME', activation = tf.nn.relu) 
			print tf.shape(layer1_out).eval()
			layer2_out = tf.layers.conv2d(layer1_out, filters = 32, kernel_size = [4, 4], strides = [2, 2], padding = 'SAME', activation = tf.nn.relu) 
			print tf.shape(layer2_out).eval()
			layer2_shape = np.prod(np.shape(layer2_out)[1:])
			layer3_out = tf.layers.dense(tf.reshape(layer2_out, [-1,layer2_shape]), 256, activation = tf.nn.relu) 
			print tf.shape(layer3_out).eval()
			print tf.shape(tf.layers.dense(layer3_out, self.action_size, activation=None)).eval()
			return tf.layers.dense(layer3_out, self.action_size, activation=None)
			# layer1_out = tf.nn.relu(tf.nn.conv2d(input, filter=[8,8,4,16], strides=[1,4,4,1], padding='SAME')) # CNN (96x96x4) => 16*(8x8) w/ stride 4
			# layer2_out = tf.nn.relu(tf.nn.conv2d(layer1_out, filter=[4,4,16,32], strides=[1,2,2,1], padding='SAME')) # CNN (23x23x16) => 32*(4x4) w/ stride 2
			# layer3_out = tf.dense(layer_2_out, 256, activation='relu') # CNN (23x23x16) => 32*(4x4) w/ stride 2
			# return tf.dense(layer_3_out, self.action_size, activation=None)
			# neural_net = input
			# for n in self.layer_sizes:
			# 	neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
			# return tf.layers.dense(neural_net, self.action_size, activation=None)

	# def Qnn(self, input):
	#	 with tf.device('/device:GPU:0'):
	#		 neural_net = input
	#		 for n in self.layer_sizes:
	#			 neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
	#		 return tf.layers.dense(neural_net, self.action_size, activation=None)

	def map_action(self, action_index):
		s = self.steering[int(np.floor(action_index/(self.acceleration_size*self.deceleration_size)))]
		a = self.acceleration[int(np.floor(action_index/self.deceleration_size))%self.acceleration_size]
		d = self.deceleration[action_index%self.deceleration_size]
		return [s,a,d]

	def initialize_tf_variables(self):
		# Setting up game specific variables
		self.env = gym.make(self.game)
		self.state_size = self.get_state_space_size(self.env.reset())
		self.lower_bounds = self.env.observation_space.low
		self.upper_bounds = self.env.observation_space.high
		self.action_size = self.steering_size*self.acceleration_size*self.deceleration_size
		# Tf placeholders
		self.state_tf = tf.placeholder(shape=[None, 96, 96, self.history_pick], dtype=tf.float64)
		self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float64)
		self.y_tf = tf.placeholder(dtype=tf.float64)

		# Operations
		self.Q_value = self.Q_nn(self.state_tf)
		self.Q_argmax = tf.argmax(self.Q_value[0])
		self.Q_amax = tf.reduce_max(self.Q_value[0])
		self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1)
		
		# Training related
		self.loss = tf.reduce_mean(tf.square(self.y_tf - self.Q_value_at_action))
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)
		self.fixed_weights = None

		# Tensorflow session setup
		config = tf.ConfigProto()
		config.allow_soft_placement=True
		config.gpu_options.allow_growth = True
		# config.log_device_placement = True
		self.sess = tf.Session(config = config)
		self.trainable_variables = tf.trainable_variables()
		self.sess.run(tf.global_variables_initializer())
		self.sess.graph.finalize()


	def phi(self, states):
		# ret_vec = np.zeros([1, 4, 96, 96])
		hist_size = np.shape(states)[0]
		# return_size = np.amin([self.history_pick, hist_size])
		ret_vec = [states[i] for i in range(hist_size)[(hist_size - self.history_pick):]]
		ret_vec = np.stack(ret_vec, axis=2)
		return ret_vec

	def get_batch(self, replay_memory):
		mini_batch = random.sample(replay_memory, self.batch_size)
		state_batch = [data[0] for data in mini_batch]
		action_batch = [data[1] for data in mini_batch]
		reward_batch = [data[2] for data in mini_batch]
		next_state_batch = [data[3] for data in mini_batch]
		done_batch = [data[4] for data in mini_batch]
		return state_batch, action_batch, reward_batch, next_state_batch, done_batch

	def experience_replay(self, replay_memory):
		state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.get_batch(replay_memory)
		y_batch = [None] * self.batch_size
		dict = {self.state_tf: next_state_batch}
		dict.update(zip(self.trainable_variables, self.fixed_weights))
		Q_value_batch = self.sess.run(self.Q_value, feed_dict=dict)
		for i in range(self.batch_size):
			y_batch[i] = reward_batch[i] + (0 if done_batch[i] else self.gamma * np.max(Q_value_batch[i]))

		self.sess.run(self.train_op, feed_dict={self.y_tf: y_batch, self.action_tf: action_batch, self.state_tf: state_batch})

	def get_random_action(self):
		s = np.random.randint(0, self.steering_size)
		a = np.random.randint(0, self.acceleration_size)
		d = np.random.randint(0, self.deceleration_size)
		return s*self.acceleration_size*self.deceleration_size + a*self.deceleration_size + d

	def get_action(self, state, epsilon):
		if random.random() < epsilon:
			return self.get_random_action()
		else:
			return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})

	def update_fixed_weights(self):
		self.fixed_weights = self.sess.run(self.trainable_variables)

	def begin_training(self, num_episodes):
		eps_decay_rate = (self.epsilon_min - self.epsilon_max) / num_episodes
		# q_averages = np.zeros(num_episodes)
		replay_memory = []
		print 'Training...'
		for episode in range(num_episodes):
			done = False
			tot_reward = 0
			state = self.env.reset()
			state = self.down_sample(state)
			states = [state, state, state, state]
			self.env.render()
			self.update_fixed_weights()
			while not done:
				# Take action and update replay memory
				phi = self.phi(states)
				action = self.get_action(phi, self.epsilon_max + eps_decay_rate * episode)
				next_state, reward, done, _ = self.env.step(self.map_action(action))
				next_state = self.down_sample(next_state)
				states.append(next_state)
				phi_1 = self.phi(states)
				one_hot_action = np.zeros(self.action_size)
				one_hot_action[action] = 1
				replay_memory.append((phi, one_hot_action, reward, phi_1, done))

				# Check whether replay memory capacity reached
				if (len(replay_memory) > self.memory_capacity): 
					replay_memory.pop(0)

				# Perform experience replay if replay memory populated
				if len(replay_memory) > 10 * self.batch_size:
					self.experience_replay(replay_memory)

				tot_reward += reward
				state = next_state
			# q_averages[episode] = self.estimate_avg_q(1000)
			print 'Episode: {}. Reward: {}'.format(episode, tot_reward)
		# file_name = 'avg_q_' + self.game + '.csv'
		# np.savetxt(file_name, q_averages, delimiter=',')
		print '--------------- Done training ---------------'
	
	def test_Q(self, num_test_episodes):
		print 'Testing...'
		for episode in range(num_test_episodes):
			done = False
			tot_reward = 0
			state = self.env.reset()
			while not done:
				# Take action and update replay memory
				action = self.get_action(state, 0)
				next_state, reward, done, _ = self.env.step(action)
				tot_reward += reward
				state = next_state
				tot_reward += reward
			print 'Test {}: Reward = {}'.format(episode, tot_reward)

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
