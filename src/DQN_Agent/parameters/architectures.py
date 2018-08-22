import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np

########## A class storing several classes of neural network architectures ##########
########## Called during training and testing. Uses tensorflow backend     ##########

# A class that defines a fully connected network with two hideen layers, each 
# with 32 neurons and a non-linear rectifier linear unit (ReLU) activation.
class Basic_Architecture:

    def __init__(self):
        self.layer_sizes = [32, 32]

    def evaluate(self, input, action_size):
        neural_net = input
        for n in self.layer_sizes:
            neural_net = tf.layers.dense(neural_net, n, activation=tf.nn.relu)
        output = tf.layers.dense(neural_net, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "2 dense layers of size {0} and {1}".format(self.layer_sizes[0], self.layer_sizes[1])


# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 16 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 fully connected layer with 16 neurons and ReLU activation. Dropout is applied with
#   keep probability of .8
class Conv_1Layer:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=16, kernel_size=[8, 8],
                                      strides=[4, 4], padding='same', activation=tf.nn.relu, data_format='channels_first', name='layer1_out')
        layer1_shape = np.prod(np.shape(layer1_out)[1:])
        layer2_out = tf.nn.dropout(tf.layers.dense(tf.reshape(layer1_out, [-1, layer1_shape]), 16, activation=tf.nn.relu), .8, name='layer2_out')
        output = tf.layers.dense(layer2_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "1 convolutional layer: filters = 16, kernel = [8,8], strides = [4,4], relu activation and one dense dropout layer with drop probability 20\% and 16 neurons"


# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 16 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 32 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 256 neurons and ReLU activation. Dropout is applied with
#   keep probability of .7
class Conv_2Layer:

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


# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 16 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 32 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 256 neurons and ReLU activation. 
# Based on 2013 paper 'Playing Atari with Deep Reinforcement Learning' by Mnih et al
class Atari_Paper:

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

# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 convolutional layer with 64 3x3 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 512 neurons and ReLU activation. 
# Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
class Nature_Paper:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out')
        layer2_out = tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu, name='layer4_out')
        output =  tf.layers.dense(layer4_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015"

# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 convolutional layer with 64 3x3 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 512 neurons and ReLU activation. 
# - Same as 'Nature_Paper' but with batchnorm on output layer
# Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
class Nature_Paper_Batchnorm:

    def evaluate(self, input, action_size):
        layer1_out = tf.contrib.layers.batch_norm(tf.layers.conv2d(input, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out'), data_format='NCHW')
        layer2_out = tf.contrib.layers.batch_norm(tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out'), data_format='NCHW')
        layer3_out = tf.contrib.layers.batch_norm(tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out'), data_format='NCHW')
        layer4_out = tf.contrib.layers.batch_norm(tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu, name='layer4_out'))
        output =  tf.contrib.layers.batch_norm(tf.layers.dense(layer4_out, action_size, activation=None, name='output'))
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015 with batchnorm on every layer"

# A class that defines a neural network with the following architecture:
# - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
# - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
# - 1 convolutional layer with 64 3x3 kernels with a stride of 2x2 w/ ReLU activation
# - 1 fully connected layer with 512 neurons and ReLU activation. 
# - Same as 'Nature_Paper' but with dropout with keep probability .7 on output layer
# Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
class Nature_Paper_Dropout:

    def evaluate(self, input, action_size):
        layer1_out = tf.layers.conv2d(input, filters=32, kernel_size=[8,8],
            strides=[4,4], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer1_out')
        layer2_out = tf.layers.conv2d(layer1_out, filters=64, kernel_size=[4,4],
            strides=[2,2], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.nn.dropout(tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu), .7, name='layer4_out')
        output =  tf.layers.dense(layer4_out, action_size, activation=None, name='output')
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015 with dropout on dense layers, 0.7 keep prob."

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
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer2_out'), .5, name='layer2_out')
        layer3_out = tf.layers.conv2d(layer2_out, filters=64, kernel_size=[3,3],
            strides=[1,1], padding='same', activation=tf.nn.relu, data_format='channels_first', 
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='layer3_out')
        layer4_out = tf.layers.dense(tf.layers.flatten(layer3_out), 512, activation=tf.nn.relu)
        output =  tf.layers.dense(layer4_out, action_size, activation=None)
        return output

    def __str__(self):
        return "Architecture used in the nature paper in 2015 with dropout on 2nd conv layer, 0.7 keep prob."