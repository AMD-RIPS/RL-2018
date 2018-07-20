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
import time
from tensorflow.python.saved_model import tag_constants

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


class DQN_Agent:
    # architecture, explore_rate and learning_rate are strings, see respective files for definitions
    def __init__(self, environment, architecture, explore_rate, learning_rate):
        self.env = environment
        self.architecture = arch.arch_dict[architecture]
        self.explore_rate = expl.expl_dict[explore_rate]()
        self.learning_rate = lrng.lrng_dict[learning_rate]()
        self.initialize_tf_variables()

    def set_training_parameters(self, discount, batch_size, memory_capacity, num_episodes):
        self.discount = discount
        self.replay_memory = rplm.Replay_Memory(memory_capacity, batch_size)
        self.num_episodes = num_episodes

    def initialize_tf_variables(self):
        # Setting up game specific variables
        self.state_size = self.env.state_space_size
        self.action_size = self.env.action_space_size
        self.state_shape = self.env.state_shape
        self.q_grid = None

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32)
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.y_tf = tf.placeholder(dtype=tf.float32)
        self.alpha = tf.placeholder(dtype=tf.float32)
        self.training_score = tf.placeholder(dtype=tf.float32)
        self.avg_q = tf.placeholder(dtype=tf.float32)
        self.epsilon = tf.placeholder(dtype=tf.float32)

        with tf.device('/device:GPU:0'):
        	# Operations
            self.Q_value = self.architecture(self.state_tf, self.action_size)
            self.Q_argmax = tf.argmax(self.Q_value[0])
            self.Q_amax = tf.reduce_max(self.Q_value[0])
            self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1)

            # Training related
            self.loss = tf.reduce_mean(tf.square(self.y_tf - self.Q_value_at_action))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss)
            self.fixed_target_weights = None

        # Tensorflow session setup
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.trainable_variables = tf.trainable_variables()

        # Tensorboard setup
        LOG_PATH = DIR_PATH  + "/logs/tmp"
        self.writer = tf.summary.FileWriter(LOG_PATH)
        self.writer.add_graph(self.sess.graph)
        training_score = tf.summary.scalar("Training score", self.training_score, collections=None, family=None)
        epsilon = tf.summary.scalar("Epsilon", self.epsilon, collections=None, family=None)
        avg_q = tf.summary.scalar("Average Q-value", self.avg_q, collections=None, family=None)
        self.training_summary = tf.summary.merge([avg_q, epsilon])
        self.test_summary = tf.summary.merge([training_score])
        subprocess.Popen(['tensorboard', '--logdir', DIR_PATH  + "/logs/tmp", '--port', '6006'])

        # Initialising saver
        self.saver = tf.train.Saver()

        # Initialising and finalising
        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()


    def experience_replay(self, alpha):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_batch()
        y_batch = [None] * self.replay_memory.batch_size

        feed_dict = {self.state_tf: next_state_batch}
        feed_dict.update(zip(self.trainable_variables, self.fixed_target_weights))
        
        Q_value_batch = self.sess.run(self.Q_value, feed_dict=feed_dict)
        y_batch = reward_batch + self.discount*np.multiply(np.invert(done_batch), np.amax(Q_value_batch, axis=1))

        # Performing one step of optimization
        self.sess.run(self.train_op, feed_dict={self.y_tf: y_batch, self.action_tf: action_batch, self.state_tf: state_batch, self.alpha: alpha})

    def get_action(self, state, epsilon):
        # Perorming epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})

    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

    def train(self):
        frame = 0
        for episode in range(self.num_episodes):
            start_time = time.time()
            print('Episode {0}/{1}'.format(episode, self.num_episodes))

            # Setting up game environment
            state = self.env.reset()
            state = self.env.process(state)
            self.env.render()

            # Setting up parameters for the episode
            done = False
            epsilon = self.explore_rate.get(frame)
            alpha = self.learning_rate.get(episode, self.num_episodes)
            while not done:
                # Updating fixed target weights every 1000 frames
                if frame % 1000 == 0:
                    self.update_fixed_target_weights()
                frame += 1

                # Choosing and performing action and updating the replay memory
                action = self.get_action(state, epsilon) 
                next_state, reward, done, info = self.env.step(action)
                next_state = self.env.process(next_state)
                reward  = np.sign(reward)
                self.replay_memory.add(self.env, state, action, reward, next_state, done, self.action_size)

                # Performing experience replay if replay memory populated
                if self.replay_memory.length() > self.replay_memory.batch_size:
                    self.experience_replay(alpha)
                state = next_state
                done = info['true_done']

            
            # Creating q_grid if not yet defined and calculating average q-value
            if not self.q_grid and self.replay_memory.length() > 5000:
                self.q_grid = self.replay_memory.get_q_grid(1000)
            avg_q = self.estimate_avg_q()
            
            # Saving tensorboard data
            if (episode % 30 == 0) and (episode != 0):
                score = self.test_Q(num_test_episodes=5, visualize = True)
                self.writer.add_summary(self.sess.run(self.test_summary, feed_dict={self.training_score: score}), episode/30)
            self.writer.add_summary(self.sess.run(self.training_summary, feed_dict={self.avg_q: avg_q, self.epsilon: epsilon}), episode)
            
            # Saving model
            if episode % 100 == 0:
                self.saver.save(self.sess, DIR_PATH + '/saved_models/tmp/data.chkp')

    def test_Q(self, num_test_episodes=10, visualize=False):
        cum_reward = 0
        for episode in range(num_test_episodes):
            done = False
            start_time = time.time()
            elapsed_time = 0
            state = self.env.reset()
            while not done and not elapsed_time > 60:
                if visualize:
                    self.env.render()
                action = self.get_action(self.env.process(state), epsilon=0)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                cum_reward += reward
                done = info['true_done']
                elapsed_time = time.time() - start_time
        return cum_reward / float(num_test_episodes)

    def estimate_avg_q(self):
        if not self.q_grid:
            return 0
        num_samples = len(self.q_grid)
        return np.average(np.amax(self.sess.run(self.Q_value, feed_dict={self.state_tf: self.q_grid}), axis = 1))