import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np


class basic_architecture:

    def __init__(self):
        self.layer_sizes = [32, 32]

    def evaluate(self, input, action_size):
        neural_net = input
        for n in self.layer_sizes:
            neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
        output = tf.layers.dense(neural_net, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "2 dense layers of size {0} and {1}".format(basic_layer_sizes[0], basic_layer_sizes[1])


class convolutional_architecture_1_layer:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=16, kernel_size=[8, 8],
                                      strides=[4, 4], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer1_out')
        layer1_shape = np.prod(np.shape(layer1_out)[1:])
        layer2_out = tf.nn.dropout(tf.layers.dense(tf.reshape(layer1_out, [-1, layer1_shape]), 16, activation=tf.nn.relu), .8, name='layer2_out')
        output = tf.layers.dense(layer2_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "1 convolutional layer: filters = 16, kernel = [8,8], strides = [4,4], relu activation and one dense dropout layer with drop probability 20\% and 16 neurons"


class convolutional_architecture_2_layers:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=16, kernel_size=[8, 8],
                                      strides=[4, 4], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer1_out')
        layer2_out = tf.layers.conv2d(layer1_out, filters=32, kernel_size=[4, 4],
                                      strides=[2, 2], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer2_out')
        layer2_out = tf.layers.flatten(layer2_out)
        layer3_out = tf.nn.dropout(tf.layers.dense(layer2_out, 256, activation=tf.nn.relu), 0.7, name='layer3_out')
        output = tf.layers.dense(layer3_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "conv2 with dropout 0.7"


class atari_paper:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=16, kernel_size=[8, 8],
                                      strides=[4, 4], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer1_out')
        layer2_out = tf.layers.conv2d(layer1_out, filters=32, kernel_size=[4, 4],
                                      strides=[2, 2], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer2_out')
        layer3_out = tf.layers.dense(tf.layers.flatten(layer2_out), 256, activation=tf.nn.relu, name='layer3_out')
        output = tf.layers.dense(layer3_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "Architecture used in the atari paper"

arch_dict = {
    'basic': basic_architecture,
    'conv1': convolutional_architecture_1_layer,
    'conv2': convolutional_architecture_2_layers,
    'atari': atari_paper
}
