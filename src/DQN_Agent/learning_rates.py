import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return 0.005

    def __str__(self):
        return '0.005'


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
