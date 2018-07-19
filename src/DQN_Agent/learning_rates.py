import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, episode, num_episodes):
        return 0.001

class Atari:

    def __init__(self):
        pass

    def get(self, episode, num_episodes):
        return 0.0005        

lrng_dict = {
    'basic': Basic, 
    'atari': Atari
}
