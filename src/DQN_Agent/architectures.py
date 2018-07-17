import sys
sys.dont_write_bytecode = True

import tensorflow as tf

basic_layer_sizes = [16, 16]

def basic_architecture(input, action_size):
    neural_net = input
    for n in basic_layer_sizes:
        neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
    return tf.layers.dense(neural_net, action_size, activation=None)


def convolutional_architecture(input, action_size):
    neural_net = input
    for n in basic_layer_sizes:
        neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
    return tf.layers.dense(neural_net, action_size, activation=None)

arch_dict = {
    'basic': basic_architecture, 
    'conv': convolutional_architecture
}

