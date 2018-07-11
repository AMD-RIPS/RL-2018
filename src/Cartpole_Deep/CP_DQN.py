import gym
import numpy as np
import random
import tensorflow as tf
from DNN import DNN

class DQN:

	# Hyperparameters for RL
	N = 10000		# replay memory capacity
	D = []			# replay memory
	NUM_EPISODES = 1000
	MAX_STEPS = 200
	MAX_EPSILON = 1
	MIN_EPSILON	= 0.01
	DECAY_RATE = 0.0001 
	DISCOUNT = 0.99
	LEARNING_RATE = 0.001
	MINIBATCH_SIZE = 32 
	TARGET_UPDATE_FREQ = 10
	LOG_DIR = '/home/jguan/RIPSAMD/src/Cartpole_Deep/log'

	def __init__(self, env):
		self.env = gym.make(env)
		self.input_size = self.env.observation_space.shape[0]
		self.output_size = self.env.action_space.n
		self.DNN = DNN(self.input_size, self.output_size, self.LEARNING_RATE)

	def get_epsilon(self, step):
		return self.MIN_EPSILON + (self.MAX_EPSILON - self.MIN_EPSILON)*np.exp(-self.DECAY_RATE*step)

	def get_action(self, state, epsilon):
		if (np.random.uniform() < epsilon):
			return np.random.randint(0, self.output_size)
		else:
			return np.asscalar(self.get_opt_action(state))

	def get_opt_action(self, state):
		feed_dict={self.DNN.x: [state]}
		q_value = self.get_q_value(feed_dict)
		return q_value.argmax(axis=1)

	def get_q_value(self, feed_dict):
		return self.sess.run(self.DNN.Q, feed_dict=feed_dict)

	def get_max_q(self, feed_dict):
		q_values = self.get_q_value(feed_dict)
		return q_values.max(axis=1)

	def get_y(self, minibatch, max_q_values):
		target_q = np.zeros(self.MINIBATCH_SIZE)
		target_action_mask = np.zeros((self.MINIBATCH_SIZE, 
			self.output_size), dtype=int)
		for i in range(self.MINIBATCH_SIZE):
			_, action, reward, _, terminal = minibatch[i]
			target_q[i] = reward
			if not terminal:
				target_q[i] += self.DISCOUNT * max_q_values[i]
			target_action_mask[i][action] = 1
		return target_q, target_action_mask

	def fillD(self):
		state = self.env.reset()
		for ii in range(self.MINIBATCH_SIZE+1):
			action = self.env.action_space.sample()
			obs, reward, done, _ = self.env.step(action)

			if done:
				# The simulation fails so no next state
				obs = np.zeros(state.shape)
				# Add experience to memory
				self.D.append((state, action, reward, obs, done))
				# Start new episode
				self.env.reset()
				# Take one random step to get the pole and cart moving
				state, reward, done, _ = self.env.step(self.env.action_space.sample())

			else:
				# Add experience to memory
				self.D.append((state, action, reward, obs, done))
				state = obs

	def train(self, num_episodes=NUM_EPISODES):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.DNN.init_op)
		tf.summary.scalar('loss', self.DNN.loss)
		self.summary = tf.summary.merge_all()
		self.summary_writer = tf.summary.FileWriter(self.LOG_DIR, self.sess.graph)

		self.fillD()
		total_steps = 0
		step_counts = []


		target_weights = self.sess.run(self.DNN.weights)
		for episode in range(num_episodes):
			state = self.env.reset()
			done = False
			tot_reward = 0
			steps = 0

			for step in range(self.MAX_STEPS):
				action = self.get_action(state, self.get_epsilon(total_steps))
				obs, reward, done, _ = self.env.step(action)

				if done:
					reward = 0
					obs = np.zeros(state.shape)
					
				tot_reward += reward

				self.D.append((state, action, reward, obs, done))
				if len(self.D) > self.N:
					self.D.pop(0)
				state = obs

				minibatch = random.sample(self.D, self.MINIBATCH_SIZE)
				next_states = [m[3] for m in minibatch]
				# TODO: Optimize to skip terminal states
				feed_dict = {self.DNN.x: next_states}
				feed_dict.update(zip(self.DNN.weights, target_weights))
				max_q_values = self.get_max_q(feed_dict)
				states = [m[0] for m in minibatch]
				target_q, target_action_mask = self.get_y(minibatch, max_q_values)
				feed_dict = {
					self.DNN.x: states, 
					self.DNN.targetQ: target_q,
					self.DNN.targetActionMask: target_action_mask,
				}
				_, summary = self.sess.run([self.DNN.train_op, self.summary], 
					feed_dict=feed_dict)

				if total_steps % 100 == 0:
					self.summary_writer.add_summary(summary, total_steps)

				total_steps += 1
				steps += 1
					
				if done:
					print('Episode: {}'.format(episode),
						'Total reward: {}'.format(tot_reward),
						'Explore P: {:.4f}'.format(self.get_epsilon(total_steps)))
					break

			# step_counts.append(steps) 
			# mean_steps = np.mean(step_counts[-100:])
			# print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}"
			# 	.format(episode, total_steps, mean_steps))

			if episode % self.TARGET_UPDATE_FREQ == 0:
				target_weights = self.sess.run(self.DNN.weights)


	def play(self):
		state = self.env.reset()
		done = False
		steps = 0
		while not done and steps < 200:
				self.env.render()
				feed_dict={self.DNN.x: [state]}
				q_values = self.get_q_value(feed_dict)
				action = q_values.argmax()
				state, _, done, _ = self.env.step(action)
				steps += 1
		return steps


if __name__ == '__main__':
	dqn = DQN('CartPole-v0')

	dqn.train()

	res = []
	for i in range(100):
		steps = dqn.play()
		print("Test steps = ", steps)
		res.append(steps)
	print("Mean steps = ", sum(res) / len(res))

	dqn.sess.close()