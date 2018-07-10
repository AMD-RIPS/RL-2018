import gym
import numpy as np
import tensorflow as tf
from DNN import DNN

class DQN:

	# Hyperparameters for RL
	N = 300		# replay memory capacity
	D = []			# replay memory
	NUM_EPISODES = 100
	EPSILON = .3
	DISCOUNT = 0.99
	LEARNING_RATE = 0.05
	MINIBATCH_SIZE = 30
	TARGET_UPDATE_FREQ = 10

	def __init__(self, env):
		self.env = gym.make(env)
		self.input_size = self.env.observation_space.shape[0]
		self.output_size = self.env.action_space.n
		self.DNN = DNN(self.input_size, self.output_size, self.LEARNING_RATE)

	def get_epsilon(self, episode):
		return EPSILON

	def get_action(self, state, epsilon):
		if (np.random.uniform() < epsilon):
			return np.random.randint(0, self.output_size)
		else:
			return get_opt_action(state)

	def get_opt_action(self, state):
		feed_dict={self.DNN.x: [state]}
		q_value = get_q_value(feed_dict)
		return tf.argmax(q_value, axis=1)[0]

	def get_q_value(self, feed_dict):
		return self.sess.run(self.DNN.Q, feed_dict=feed_dict)

	def get_max_q(self, feed_dict):
		q_values = get_q_value(feed_dict)
		return q_values.max(axis=1)

	def get_y(self, max_q_values):
		target_q = np.zeros(self.MINIBATCH_SIZE)
		target_action_mask = np.zeros((self.MINIBATCH_SIZE, 
			self.output_size), dtype=int)
		for i in range(self.MINIBATCH_SIZE):
			_, action, reward, _, terminal = minibatch[i]
			target_q[i] = reward
			if not terminal:
				target_q[i] += self.DISCOUNT_FACTOR * max_q_values[i]
			target_action_mask[i][action] = 1
		return target_q, target_action_mask

	def train(self, num_episodes=NUM_EPISODES):
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.DNN.init_op)
		total_steps = 0

		target_weights = self.sess.run(self.DNN.weights)
		for episode in range(num_episodes):
			state = self.env.reset()
			done = False
			tot_reward = 0

			while not done:
				action = get_action(state, get_epsilon(episode))
				obs, reward, done, _ = self.env.step(action)

				self.D.append((state, action, reward, obs, done))
				if len(self.D) > N:
					self.D.pop(0)
				state = obs

				if len(self.D) >= self.MINIBATCH_SIZE:
					minibatch = random.sample(self.D, self.MINIBATCH_SIZE)
					next_states = [m[3] for m in minibatch]
          			# TODO: Optimize to skip terminal states
					feed_dict = {self.DNN.x: next_states}
					feed_dict.update(zip(self.weights, target_weights))
					max_q_values = get_max_q(feed_dict)
					states = [m[0] for m in minibatch]
					target_q, target_action_mask = get_y(max_q_values)
					feed_dict = {
						self.x: states, 
						self.targetQ: target_q,
						self.targetActionMask: target_action_mask,
					}
					_, summary = self.sess.run([self.DNN.train_op, self.DNN.summary], 
						feed_dict=feed_dict)
				total_steps += 1

			if episode % self.TARGET_UPDATE_FREQ == 0:
				target_weights = self.sess.run(self.DNN.weights)
		sess.close()

	def play(self):
		state = self.env.reset()
		done = False
		steps = 0
		while not done and steps < 200:
				self.env.render()
				feed_dict={self.DNN.x: [state]}
				q_values = get_q_value(feed_dict)
				action = q_values.argmax()
				state, _, done, _ = self.env.step(action)
				steps += 1
		return steps


if __name__ == '__main__':
	dqn = DQN('CartPole-v0')

	dqn.env.monitor.start('/tmp/cartpole', force=True)
	dqn.train()
	dqn.env.monitor.close()

	res = []
	for i in range(100):
		steps = dqn.play()
		print("Test steps = ", steps)
		res.append(steps)
	print("Mean steps = ", sum(res) / len(res))