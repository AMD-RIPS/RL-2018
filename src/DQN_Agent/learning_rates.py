import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return max(0.005, 0.05* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))

    def __str__(self):
        return 'max(0.005, 0.05* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))'


class Atari:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return 0.00001

    def __str__(self):
        return '0.00001'

lrng_dict = {
    'basic': Basic,
    'atari': Atari
}
