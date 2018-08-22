class Decay_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))'

class Basic_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, 0.5* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))

    def __str__(self):
        return 'max(0.1, 0.5* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))'

class Atari_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))'

class Fixed_Explore_Rate:

    def get(self, training_metadata):
        return 0.1

    def __str__(self):
        return '0.1'