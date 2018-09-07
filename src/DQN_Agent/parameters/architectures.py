import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np

# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 convolutional layer with 64 3x3 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 512 neurons and ReLU activation. 
# - Same as 'Nature_Paper' but with dropout with keep probability .5 on 2nd convolutional layer
# Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
class Nature_Paper_Conv_Dropout:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out')
        layer2_out = tf.nn.dropout(tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out'), 1, name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu)
        output =  tf.layers.dense(layer4_out, action_size, activation=None)
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015 with dropout on 2nd conv layer, 0.5 keep prob."