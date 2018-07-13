import gym
import numpy as np
import tensorflow as tf
import random
import os 
import subprocess
DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class Playground:

    def __init__(self, game, num_hidden_layers, layer_sizes, epsilon_max, epsilon_min, alpha, gamma, batch_size, memory_capacity):
        self.game = game
        self.num_hidden_layers = num_hidden_layers
        self.layer_sizes = layer_sizes
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.initialize_tf_variables()

    def Q_nn(self, input):
        with tf.device('/device:CPU:0'):
            neural_net = input
            for n in self.layer_sizes:
                neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
            return tf.layers.dense(neural_net, self.action_size, activation=None)

    def initialize_tf_variables(self):
        # Setting up game specific variables
        self.env = gym.make(self.game)
        self.state_size = np.shape(self.env.observation_space)[0]
        self.lower_bounds = self.env.observation_space.low
        self.upper_bounds = self.env.observation_space.high
        self.action_size = self.env.action_space.n
        self.q_grid = None

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=[None, self.state_size], dtype=tf.float64)
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float64)
        self.y_tf = tf.placeholder(dtype=tf.float64)
        self.training_score = tf.placeholder(dtype=tf.float64)
        self.avg_q = tf.placeholder(dtype=tf.float64)

        # Operations
        self.Q_value = self.Q_nn(self.state_tf)
        self.Q_argmax = tf.argmax(self.Q_value[0])
        self.Q_amax = tf.reduce_max(self.Q_value[0])
        self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1)

        # Training related
        self.loss = tf.reduce_mean(tf.square(self.y_tf - self.Q_value_at_action))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)
        self.fixed_weights = None

        # Tensorflow  session setup
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        self.sess = tf.Session(config=config)
        self.trainable_variables = tf.trainable_variables()

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(DIR_PATH)
        self.writer.add_graph(self.sess.graph)
        tf.summary.scalar("Training score", self.training_score, collections=None, family=None)
        tf.summary.scalar("Average Q-value", self.avg_q, collections=None, family=None)
        self.summary = tf.summary.merge_all()
        subprocess.Popen(['tensorboard', '--logdir', DIR_PATH])

        # Initialising and finalising
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()

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

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return self.env.action_space.sample()
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
            self.update_fixed_weights()
            while not done:
                # Take action and update replay memory
                action = self.get_action(state, self.epsilon_max + eps_decay_rate * episode)
                next_state, reward, done, _ = self.env.step(action)
                one_hot_action = np.zeros(self.action_size)
                one_hot_action[action] = 1
                replay_memory.append((state, one_hot_action, reward, next_state, done))

                # Check whether replay memory capacity reached
                if (len(replay_memory) > self.memory_capacity):
                    replay_memory.pop(0)

                # Perform experience replay if replay memory populated
                if len(replay_memory) > 10 * self.batch_size:
                    self.experience_replay(replay_memory)

                tot_reward += reward
                state = next_state
            if len(replay_memory) > 1000:
                self.q_grid = [[replay_memory[index][0]] for index in range(1000)]
            avg_q = self.estimate_avg_q(1000)
            score = self.test_Q(num_test_episodes=5)
            self.writer.add_summary(self.sess.run(self.summary, feed_dict={self.training_score:score, self.avg_q:avg_q}), episode)
            print 'Episode: {}. Reward: {}'.format(episode, score)
        print '--------------- Done training ---------------'

    def test_Q(self, num_test_episodes = 10, visualize=False):
        cum_reward = 0
        for episode in range(num_test_episodes):
            done = False
            state = self.env.reset()
            while not done:
                if visualize:
                    self.env.render()
                action = self.get_action(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                cum_reward += reward
        return cum_reward / float(num_test_episodes)

    def estimate_avg_q(self, num_samples):
        if not self.q_grid: return 0
        q_avg = 0.0
        for index in range(num_samples):
            q_avg += np.amax(self.sess.run(self.Q_value, feed_dict={self.state_tf: self.q_grid[index]}))
        return q_avg/num_samples
