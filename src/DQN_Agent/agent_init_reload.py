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
from utils import pause
import time
from tensorflow.python.saved_model import tag_constants

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH = DIR_PATH + '/saved_models/tmp3'
LOG_PATH = DIR_PATH  + "/logs/tmp3"

class DQN_Agent:
    # architecture, explore_rate and learning_rate are strings, see respective files for definitions
    def __init__(self, environment, architecture, explore_rate, learning_rate, reload_bool):
        self.env = environment
        self.architecture = arch.arch_dict[architecture]
        self.explore_rate = expl.expl_dict[explore_rate]()
        self.learning_rate = lrng.lrng_dict[learning_rate]()
        if reload_bool:
            self.reload_tf_variables()
        else:
            self.initialize_tf_variables()

    def set_training_parameters(self, discount, batch_size, memory_capacity, num_episodes, 
        learning_rate_drop_frame_limit = 100000, test_frequency = 100, save_frequence = 50, 
        num_q_grid = 1000):
        self.discount = discount
        self.replay_memory = rplm.Replay_Memory(memory_capacity, batch_size)
        self.num_episodes = num_episodes
        self.training_metadata = {'num_episodes': num_episodes, 'frame_limit': learning_rate_drop_frame_limit}
        self.test_frequency = test_frequency
        self.save_frequence = save_frequence
        self.num_q_grid = num_q_grid

    def initialize_tf_variables(self):
        # Setting up game specific variables
        self.state_size = self.env.state_space_size
        self.action_size = self.env.action_space_size
        self.state_shape = self.env.state_shape

        # Tf placeholders
        self.state_tf = tf.placeholder(shape=self.state_shape, dtype=tf.float32, name='state_tf')
        self.action_tf = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='action_tf')
        self.y_tf = tf.placeholder(dtype=tf.float32, name='y_tf')
        self.alpha = tf.placeholder(dtype=tf.float32, name='alpha')
        self.training_score = tf.placeholder(dtype=tf.float32, name='training_score')
        self.avg_q = tf.placeholder(dtype=tf.float32, name='avg_q')
        self.loss_monitor = tf.placeholder(dtype=tf.float32, name='loss_monitor')

        self.q_grid_set = tf.Variable(initial_value = tf.random_uniform([1000,4]), dtype = tf.float32, name = "q_grid_set")
        self.q_stored = False

        self.epsilon = tf.placeholder(dtype=tf.float32, name='epsilon')
        self.episode = tf.Variable(initial_value=0,trainable=False, name='episode')
        self.frames = tf.Variable(initial_value=0,trainable=False, name='frames')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.increment_global_step_op = tf.assign(self.global_step, self.global_step+1, name='increment_global_step_op')
        self.increment_frames_op = tf.assign(self.frames, self.frames+1, name='increment_frames_op')
        self.increment_episode_op = tf.assign(self.episode, self.episode+1, name='increment_episode_op')

        # Operations
        self.Q_value = self.architecture(self.state_tf, self.action_size)
        self.Q_argmax = tf.argmax(self.Q_value[0], name='Q_argmax')
        self.Q_amax = tf.reduce_max(self.Q_value[0], name='Q_amax')
        self.Q_value_at_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_tf), axis=1, name='Q_value_at_action')

        # Training related
        self.loss = tf.reduce_mean(tf.square(self.y_tf - self.Q_value_at_action), name='loss')
        tf.losses.add_loss(self.loss, loss_collection=tf.GraphKeys.LOSSES)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.alpha).minimize(self.loss, global_step=self.global_step, name='train_minimize')
        self.fixed_target_weights = None

        # Tensorflow session setup
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)
        self.trainable_variables = tf.trainable_variables()

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(LOG_PATH)
        self.writer.add_graph(self.sess.graph)
        training_score = tf.summary.scalar("Training score", self.training_score, collections=None, family=None)
        epsilon = tf.summary.scalar("Epsilon", self.epsilon, collections=None, family=None)
        avg_q = tf.summary.scalar("Average Q-value", self.avg_q, collections=None, family=None)
        loss_monitor = tf.summary.scalar("Loss", self.loss_monitor, collections=None, family=None)
        self.training_summary = tf.summary.merge([avg_q, epsilon])
        self.test_summary = tf.summary.merge([training_score])
        self.loss_summary = tf.summary.merge([loss_monitor])
        subprocess.Popen(['tensorboard', '--logdir', LOG_PATH, '--port', '6006'])

        # Initialising saver
        self.saver = tf.train.Saver()

        # Initialising and finalising
        self.sess.run(tf.global_variables_initializer())
        # self.sess.graph.finalize()

        self.previous_eps_num = self.sess.run(self.episode)
        self.previous_fra_num = self.sess.run(self.frames)


    def reload_tf_variables(self):
        tf.reset_default_graph() 
        self.saver = tf.train.import_meta_graph(SAVE_PATH + '/data.chkp.meta')

        # Setting up game specific variables
        self.state_size = self.env.state_space_size
        self.action_size = self.env.action_space_size
        self.state_shape = self.env.state_shape

        # Tensorflow session setup
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        self.sess = tf.Session(config=config)

        self.saver.restore(self.sess,tf.train.latest_checkpoint(SAVE_PATH))
        graph = tf.get_default_graph()

        # Tf placeholders
        self.state_tf = graph.get_tensor_by_name("state_tf:0")
        self.action_tf = graph.get_tensor_by_name("action_tf:0")
        self.y_tf = graph.get_tensor_by_name("y_tf:0")
        self.alpha = graph.get_tensor_by_name("alpha:0")
        self.training_score = graph.get_tensor_by_name("training_score:0")
        self.avg_q = graph.get_tensor_by_name("avg_q:0")
        self.epsilon = graph.get_tensor_by_name("epsilon:0")
        self.loss_monitor = graph.get_tensor_by_name("loss_monitor:0")

        self.q_grid_set = graph.get_tensor_by_name("q_grid_set:0")
        self.q_stored = True

        self.episode = graph.get_tensor_by_name("episode:0")
        self.frames = graph.get_tensor_by_name("frames:0")
        self.global_step = graph.get_tensor_by_name("global_step:0")
        self.increment_global_step_op = graph.get_tensor_by_name("increment_global_step_op:0")
        self.increment_frames_op = graph.get_tensor_by_name("increment_frames_op:0")
        self.increment_episode_op = graph.get_tensor_by_name("increment_episode_op:0")

        # Operations
        self.Q_value = graph.get_tensor_by_name("output/BiasAdd:0")
        self.Q_argmax = graph.get_tensor_by_name("Q_argmax:0")
        self.Q_amax = graph.get_tensor_by_name("Q_amax:0")
        self.Q_value_at_action = graph.get_tensor_by_name("Q_value_at_action:0")

        # # Training related
        self.loss = graph.get_tensor_by_name("loss:0")
        self.train_op = graph.get_operation_by_name("train_minimize")
        self.trainable_variables = tf.trainable_variables()
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

        self.previous_eps_num = self.sess.run(self.episode)
        self.previous_fra_num = self.sess.run(self.frames)

        # Tensorboard setup
        self.writer = tf.summary.FileWriter(LOG_PATH)
        self.writer.add_graph(self.sess.graph)
        training_score = graph.get_tensor_by_name('Training_score_1:0')
        epsilon = graph.get_tensor_by_name('Epsilon_1:0')
        avg_q = graph.get_tensor_by_name('Average_Q-value:0')
        loss_monitor = graph.get_tensor_by_name('Loss_1:0')
        self.training_summary = tf.summary.merge([avg_q, epsilon])
        self.test_summary = tf.summary.merge([training_score])
        self.loss_summary = tf.summary.merge([loss_monitor])
        subprocess.Popen(['tensorboard', '--logdir', LOG_PATH, '--port', '6007'])

        # Initialising and finalising
        # self.sess.graph.finalize()


    def experience_replay(self, alpha):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_memory.get_mini_batch()
        y_batch = [None] * self.replay_memory.batch_size

        feed_dict = {self.state_tf: next_state_batch}
        feed_dict.update(zip(self.trainable_variables, self.fixed_target_weights))
        
        Q_value_batch = self.sess.run(self.Q_value, feed_dict=feed_dict)
        Q_value_at_action = np.sum(np.multiply(action_batch, Q_value_batch), axis=1)
        y_batch = reward_batch + self.discount*np.multiply(np.invert(done_batch), np.amax(Q_value_batch, axis=1))

        loss = self.sess.run(self.loss, feed_dict={self.y_tf: y_batch, self.Q_value_at_action: Q_value_at_action})

        # Performing one step of optimization
        self.sess.run([self.train_op, self.increment_global_step_op], 
            feed_dict={self.y_tf: y_batch, self.action_tf: action_batch, 
            self.state_tf: state_batch, self.alpha: alpha})
        self.writer.add_summary(self.sess.run(self.loss_summary, 
            feed_dict={self.loss_monitor: loss}), self.sess.run(self.episode))

    def get_action(self, state, epsilon):
        # Perorming epsilon-greedy action selection
        if random.random() < epsilon:
            return self.env.sample_action_space()
        else:
            return self.sess.run(self.Q_argmax, feed_dict={self.state_tf: [state]})

    def update_fixed_target_weights(self):
        self.fixed_target_weights = self.sess.run(self.trainable_variables)

    def train(self):
        self.training_metadata['frame'] = self.sess.run(self.frames)
        for episode in range(self.num_episodes):
            self.training_metadata['episode'] = self.sess.run(self.episode)
            self.sess.run(self.increment_episode_op)
            start_time = time.time()
            print('Episode {0}/{1}'.format(self.sess.run(self.episode), self.num_episodes+self.previous_eps_num))

            # Setting up game environment
            state = self.env.reset()
            state = self.env.process(state)
            self.env.render()

            # Setting up parameters for the episode
            done = False
            epsilon = self.explore_rate.get(self.training_metadata)
            alpha = self.learning_rate.get(self.training_metadata)
            while not done:
                # Updating fixed target weights every 1000 frames
                if self.training_metadata['frame'] % 1000 == 0:
                    self.update_fixed_target_weights()
                self.sess.run(self.increment_frames_op)
                self.training_metadata['frame'] = self.sess.run(self.frames)

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
            if not self.q_stored and self.replay_memory.length() > (self.num_q_grid*5):
                self.q_stored = True
                q_grid = self.replay_memory.get_q_grid(self.num_q_grid)
                update = tf.assign(self.q_grid_set, np.asarray(q_grid, dtype = np.float32))
                self.sess.run(update)
            avg_q = self.estimate_avg_q()
            
            # Saving tensorboard data
            if ((episode+1) % self.test_frequency == 0) and (episode > self.test_frequency*3):
                score = self.test_Q(num_test_episodes=5, visualize = True)
                self.writer.add_summary(self.sess.run(self.test_summary, 
                    feed_dict={self.training_score: score}), self.sess.run(self.episode)/self.test_frequency)
            self.writer.add_summary(self.sess.run(self.training_summary, 
                feed_dict={self.avg_q: avg_q, self.epsilon: epsilon}), self.sess.run(self.episode))

            # Saving model
            if (episode+1) % self.save_frequence == 0:
                self.saver.save(self.sess, SAVE_PATH + '/data.chkp')

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
                action = self.get_action(self.env.process(state), epsilon=0)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                cum_reward += reward
                done = info['true_done']
                elapsed_time = time.time() - start_time
                if elapsed_time > 60: num_test_episodes = episode
        return cum_reward / float(num_test_episodes)

    def estimate_avg_q(self):
        if not self.q_stored:
            return 0
        return np.average(np.amax(self.sess.run(self.Q_value, feed_dict={self.state_tf: self.sess.run(self.q_grid_set)}), axis = 1))