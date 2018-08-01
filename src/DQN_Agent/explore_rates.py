import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return max(0.005, 1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes)

    def __str__(self):
        return 'max(0.005, 1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes)'


class Atari:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))'

class Decay:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))'

class Fixed:

    def __init__(self):
        pass

    def get(self, training_metadata):
        return 0.1

    def __str__(self):
        return '0.1'
expl_dict = {
    'basic': Basic,
    'atari': Atari,
    'decay': Decay,
    'fixed': Fixed
}
