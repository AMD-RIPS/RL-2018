import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, episode, num_episodes):
        return max(0.01, 1 - float(episode)/((num_episodes+1)/2))

expl_dict = {
    'basic': Basic
}
