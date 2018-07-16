import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, episode, num_episodes):
        return 0.5*(1 - episode/num_episodes)

expl_dict = {
    'basic': Basic
}
