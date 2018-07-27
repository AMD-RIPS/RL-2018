import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return max(0.05, 0.8 - 2 * float(training_metadata.episode) / training_metadata.num_episodes)

    def __str__(self):
        return 'max(0.05, 0.8 - 2 * float(episode) / num_episodes)'


class Atari:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))'

expl_dict = {
    'basic': Basic,
    'atari': Atari
}
