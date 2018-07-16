import sys
sys.dont_write_bytecode = True

import tensorflow as tf


def basic_architecture(input, action_size):
    neural_net = input
    for n in basic_layer_sizes:
        neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
    return tf.layers.dense(neural_net, action_size, activation=None)

arch_dict = {
    'basic': basic_architecture
}

basic_layer_sizes = [16, 16]
