####################################################################
# This file defines the hyperarameters, environment and            #
# architecture used. To change the hyperparameters, only alter     #
# the setup_dict.                                                  #
####################################################################
import sys

sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np

# fixed_1track_seed defines the seed used to create the fixed one
# track and fixed_3track_seed defines the three seeds used to create
# the fixed three tracks environment
fixed_1track_seed = [108]
fixed_3track_seed = [104, 106, 108]

####################################################################
# Hyperparameters:												   #
# 	architecture requires a class that defines the neural network  #
# 		architecture to train on. 								   #
# 	learning_rate requires a class that defines the learning rate  #
# 	explore_rate requires a class that defines the explore rate    #
# 	target_update_frequency requires an integer that defines the   #
# 		number of frames between each target Q update;			   #
# 	batch_size requires an integer that defines the size of the    #
#		mini-batch;												   #
# 	memory_capacity requires an integer that defines the capacity  #
#		for replay memory; 										   #
# 	num_episodes requires an integer that defines the number of    #
# 		episodes the algorithm will train on before quitting;      #
# 	learning_rate_drop_frame_limit requires an integer that 	   #
# 		defines the number of frames the exploration rate decays   #
# 		over.													   #
####################################################################
# Environment:													   #
# 	seed defines the seed used for the environment, availvable     #
# 		options include:										   #
#		fixed_1track_seed (fixed one track environment),           #
# 		fixed_3track_seed (fixed three track environment) and      #
# 		None (random tracks environment) 						   #
####################################################################
# The explore rate decays from 1 to 0.1 linearly over the frame 
# limit defined in the training_metadata and stays at 0.1 thereafter
class Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))'

# The learning rate is fixed to 0.00025 constantly
class Learning_Rate:

    def get(self, training_metadata):
        return 0.00025

    def __str__(self):
        return '0.00025'

# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 convolutional layer with 64 3x3 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 512 neurons and ReLU activation. 
# - Same as 'Nature_Paper' but with dropout with keep probability .5 on 2nd convolutional layer
# Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
class Architecture:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out')
        layer2_out = tf.nn.dropout(tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out'), 0.5, name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu)
        output =  tf.layers.dense(layer4_out, action_size, activation=None)
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015 with dropout on 2nd conv layer, 0.5 keep prob."

setup_dict = {
	'agent': {
		'architecture': Architecture, 
		'learning_rate': Learning_Rate,
		'explore_rate': Explore_Rate,
		'target_update_frequency': 1000,
		'batch_size': 32, 
		'memory_capacity': 100000, 
		'num_episodes': 3000,
		'learning_rate_drop_frame_limit': 250000
	},
	'car racing': {
		'seed': fixed_3track_seed
	}
}

