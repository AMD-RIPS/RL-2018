class Atari_Learning_Rate:

    def get(self, training_metadata):
        return 0.00025

    def __str__(self):
        return '0.00025'

class Basic_Learning_Rate:

    def get(self, training_metadata):
        return max(0.001, 0.005* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))

    def __str__(self):
        return 'max(0.001, 0.005* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))'