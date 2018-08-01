import sys
sys.dont_write_bytecode = True

import gym
import numpy as np
import tensorflow as tf
import random
import os
import subprocess
import architectures as arch
import learning_rates as lrng
import explore_rates as expl
import replay_memory as rplm
import prioritized_replay_memory as prplm
import utils
import time
from tensorflow.python.saved_model import tag_constants

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class DQN_Agent:
    # architecture, explore_rate and learning_rate are strings, see respective files for definitions

    def __init__(self, environment, architecture, explore_rate, learning_rate, model_name=None):
        self.env = environment
        self.architecture = arch.arch_dict[architecture]()
        self.explore_rate = expl.expl_dict[explore_rate]()
        self.learning_rate = lrng.lrng_dict[learning_rate]()
        self.model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/' + model_name if model_name else str(self.env)
        self.log_path = self.model_path + '/log'
        self.initialize_tf_variables()

    def set_training_parameters(self, discount, batch_size, memory_capacity, num_episodes, score_limit, 
                                delta=1, learning_rate_drop_frame_limit=100000, target_update_frequency=10, replay_method='regular'):
        self.target_update_frequency = target_update_frequency
        self.discount = discount
        if replay_method == 'regular':
            self.replay_memory = rplm.Replay_Memory(memory_capacity, batch_size)
        else:
            self.replay_memory = prplm.Prioritized_Replay_Memory(memory_capacity, batch_size)
        self.training_metadata = utils.Training_Metadata(frame=self.sess.run(self.frames), frame_limit=learning_rate_drop_frame_limit,
                                                            episode=self.sess.run(self.episode), num_episodes=num_episodes)
        self.delta = delta
        self.score_limit = score_limit
        utils.document_parameters(self)

    def initialize_tf_variables(self):
        # Setting up game specific variables
        self.state_size = self.env.state_space_size
        self.action_size = self.env.action_space_size
        self.state_shape = self.env.state_shape
        self.q_grid = None

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32, name='state_tf')
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action_tf')
        self.y_tf = tf.placeholder(dtype=tf.float32, name='y_tf')
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        self.grad_weights = tf.placeholder(dtype=tf.float32, name='weighted_grads')
        self.test_score = tf.placeholder(dtype=tf.float32, name='test_score')
        self.avg_q = tf.placeholder(dtype=tf.float32, name='avg_q')

        # Keep track of episode and frames
        self.episode = tf.Variable(initial_value=0, trainable=False, name='episode')
        self.frames = tf.Variable(initial_value=0, trainable=False, name='frames')
        self.increment_frames_op = tf.assign(self.frames, self.frames + 1, name='increment_frames_op')
        self.increment_episode_op = tf.assign(self.episode, self.episode + 1, name='increment_episode_op')

        # Operations
        # NAME                      DESCRIPTION                                         FEED DEPENDENCIES
        # Q_value                   Value of Q at given state(s)                        state_tf
        # Q_argmax                  Action(s) maximizing Q at given state(s)            state_tf
        # Q_amax                    Maximal action value(s) at given state(s)           state_tf
        # Q_value_at_action         Q value at specific (action, state) pair(s)         state_tf, action_tf
        # onehot_greedy_action      One-hot encodes greedy action(s) at given state(s)  state_tf
        self.Q_value = self.architecture.evaluate(self.state_tf, self.action_size)
        self.Q_argmax = tf.argmax(self.Q_value, axis=1, name='Q_argmax')
        self.Q_amax = tf.reduce_max(self.Q_value, axis=1, name='Q_max')
        self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1, name='Q_value_at_action')
        self.onehot_greedy_action = tf.one_hot(self.Q_argmax, depth=self.action_size)

        # Training related
        # NAME                          FEED DEPENDENCIES
        # td_error                      y_tf, state_tf, action_tf
        # loss                          y_tf, state_tf, action_tf, grad_weights
        # train_op                      y_tf, state_tf, action_tf, grad_weights, alpha

        # self.loss = tf.losses.mean_squared_error(self.y_tf, self.Q_value_at_action)
        self.td_error = tf.abs(tf.subtract(self.y_tf, self.Q_value_at_action))
        self.loss = tf.multiply(self.grad_weights, tf.losses.huber_loss(self.y_tf, self.Q_value_at_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha)
        self.train_op = self.optimizer.minimize(self.loss, name='train_minimize')

        # Tensorflow session setup
        self.saver = tf.train.Saver()
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.trainable_variables = tf.trainable_variables()

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(self.log_path)
        self.writer.add_graph(self.sess.graph)
        test_score = tf.summary.scalar("Training score", self.test_score, collections=None, family=None)
        avg_q = tf.summary.scalar("Average Q-value", self.avg_q, collections=None, family=None)
        self.training_summary = tf.summary.merge([avg_q])
        self.test_summary = tf.summary.merge([test_score])
        subprocess.Popen(['tensorboard', '--logdir', self.log_path])

        # Initialising variables and finalising graph
        self.sess.run(tf.global_variables_initializer())
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

        self.sess.graph.finalize()

    def experience_replay(self, alpha):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights, indices = self.replay_memory.get_mini_batch(self.training_metadata)
        y_batch = [None] * self.replay_memory.batch_size
        fixed_feed_dict = {self.state_tf: next_state_batch}
        fixed_feed_dict.update(zip(self.trainable_variables, self.fixed_target_weights))

        # Simple DQN #########################################################################################
        # Q_batch = self.sess.run(self.Q_amax, feed_dict=fixed_feed_dict)
        ######################################################################################################

        # Double DQN #########################################################################################
        greedy_actions = self.sess.run(self.onehot_greedy_action, feed_dict={self.state_tf: next_state_batch})
        fixed_feed_dict.update({self.action_tf: greedy_actions})
        Q_batch = self.sess.run(self.Q_value_at_action, feed_dict=fixed_feed_dict)
        ######################################################################################################
        y_batch = reward_batch + self.discount * np.multiply(np.invert(done_batch), Q_batch)

        feed = {self.state_tf: state_batch, self.action_tf: action_batch, self.y_tf: y_batch, self.grad_weights: weights, self.alpha: alpha}
        new_priorities = self.sess.run(self.td_error, feed_dict=feed)
        self.replay_memory.priority_update(indices, new_priorities)
        self.sess.run(self.train_op, feed_dict=feed)

    def get_action(self, state, epsilon):
        # Perorming epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})[0]

    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

    def train(self):
        for episode in range(self.training_metadata.num_episodes):
            self.training_metadata.increment_episode()
            self.sess.run(self.increment_episode_op)

            # Setting up game environment
            state = self.env.reset()
            self.env.render()

            # Setting up parameters for the episode
            done = False
            epsilon = self.explore_rate.get(self.training_metadata)
            alpha = self.learning_rate.get(self.training_metadata)

            while not done:
                # Updating fixed target weights every #target_update_frequency frames
                if self.training_metadata.frame % self.target_update_frequency == 0 and (self.training_metadata.frame != 0):
                    self.update_fixed_target_weights()

                # Choosing and performing action and updating the replay memory
                action = self.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)

                self.replay_memory.add(self, state, action, reward, next_state, done)

                # Performing experience replay if replay memory populated
                if self.replay_memory.full():
                    self.sess.run(self.increment_frames_op)
                    self.training_metadata.increment_frame()
                    self.experience_replay(alpha)
                state = next_state
                done = info['true_done']

            # Creating q_grid if not yet defined and calculating average q-value
            if self.replay_memory.full():
                self.q_grid = self.replay_memory.get_q_grid(size=100, training_metadata=self.training_metadata)
            avg_q = self.estimate_avg_q()
            print('Score: {0},\t epsilon: {1},\t learning rate: {2}'.format(self.test_Q(5), epsilon, alpha))
            # Saving tensorboard data and model weights
            if (episode % 30 == 0) and (episode != 0):
                score = self.test_Q(num_test_episodes=5, visualize=False)
                print(score)
                self.writer.add_summary(self.sess.run(self.test_summary,
                                                      feed_dict={self.test_score: score}), episode / 30)
                self.saver.save(self.sess, self.model_path + '/data.chkp')
                if score > self.score_limit and episode > 200:
                    break

            self.writer.add_summary(self.sess.run(self.training_summary, feed_dict={self.avg_q: avg_q}), episode)

    def test_Q(self, num_test_episodes=10, visualize=False):
        cum_reward = 0
        start_time = time.time()
        elapsed_time = 0
        for episode in range(num_test_episodes):
            done = False
            state = self.env.reset()
            while not done and not elapsed_time > 60:
                if visualize:
                    self.env.render()
                action = self.get_action(state, epsilon=0)

                next_state, reward, done, info = self.env.step(action)
                state = next_state
                cum_reward += reward
                done = info['true_done']
                elapsed_time = time.time() - start_time
                if elapsed_time > 60:
                    num_test_episodes = episode
        return cum_reward / max(1, float(num_test_episodes))

    def estimate_avg_q(self):
        if not self.q_grid:
            return 0
        return np.average(np.amax(self.sess.run(self.Q_value, feed_dict={self.state_tf: self.q_grid}), axis=1))


    def load(self, path):
        self.saver.restore(self.sess, path)
