####################################################################
# This file contains two different classes for the learning rate   #
# (epsilon): Atari_Learning_Rate and Basic_Learning_Rate. Each     #
# class contains two function: get(training_metadata) for          #
# outputting the learning rate and __str__ for outputting a string #
# indicating the class used.                                       #
# The Atari_Learning_Rate class was used for the most recent tests #
####################################################################

# The learning rate is fixed to 0.00025 constantly
class Atari_Learning_Rate:

    def get(self, training_metadata):
        return 0.00025

    def __str__(self):
        return '0.00025'

# The learning rate decays from 0.005 to 0.001 linearly over the half 
# of the number of episodes defined in the training_metadata and stays 
# at 0.001 thereafter
class Basic_Learning_Rate:

    def get(self, training_metadata):
        return max(0.001, 0.005* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))

    def __str__(self):
        return 'max(0.001, 0.005* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))'