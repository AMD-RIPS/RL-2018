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
    device = 'GPU:0'
    if device[0] == 'G':
        data_format = 'channels_first'
    else:
        data_format = 'channels_last'
    with tf.device('/device:'+device):
        layer1_out = tf.layers.conv2d(input, filters=16, kernel_size=[8,8], strides=[4,4], padding='same', activation=tf.nn.relu, data_format=data_format) 
        layer1_shape = np.prod(np.shape(layer1_out)[1:])
        layer2_out = tf.nn.dropout(tf.layers.dense(tf.reshape(layer1_out, [-1,layer1_shape]), 64, activation=tf.nn.relu), .5) 
        output = tf.layers.dense(layer2_out, self.action_size, activation=None)
    return output 

arch_dict = {
    'basic': basic_architecture, 
    'conv': convolutional_architecture
}

