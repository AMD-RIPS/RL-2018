import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
basic_layer_sizes = [16, 16]

def basic_architecture(input, action_size):
    neural_net = input
    for n in basic_layer_sizes:
        neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
    return tf.layers.dense(neural_net, action_size, activation=None)


def convolutional_architecture(input, action_size):
    with tf.device('/device:GPU:0'):
        layer1_out = tf.layers.conv2d(input, filters=16, kernel_size=[8,8],
         strides=[4,4], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer1_out') # => 23x23x16
        layer1_shape = np.prod(np.shape(layer1_out)[1:])
        layer2_out = tf.nn.dropout(tf.layers.dense(tf.reshape(layer1_out, [-1,layer1_shape]),
         16, activation=tf.nn.relu), .3, name='layer2_out') # => 1x256
        output = tf.layers.dense(layer2_out, action_size, activation=None, name = 'output')
    return output

arch_dict = {
    'basic': basic_architecture, 
    'conv': convolutional_architecture
}

