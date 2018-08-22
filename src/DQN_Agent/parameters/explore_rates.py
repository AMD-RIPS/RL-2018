####################################################################
# This file contains four different classes for the explore rate   #
# (epsilon): Decay_Explore_Rate, Basic_Explore_Rate,               #
# Atari_Explore_Rate and Fixed_Explore_Rate. Each class contains   #
# two function: get(training_metadata) for outputting the explore  #
# rate and __str__ for outputting a string indicating the class    #
# used.                                                            #
# The Decay_Explore_Rate class was used for the most recent tests  # 
####################################################################

# The explore rate decays from 1 to 0.1 linearly over the frame 
# limit defined in the training_metadata and stays at 0.1 thereafter
class Decay_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))'


# The explore rate decays from 0.6 to 0.1 linearly over the half of the
# number of episodes defined in the training_metadata and stays at 0.1
# thereafter
class Basic_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, 0.5* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))

    def __str__(self):
        return 'max(0.1, 0.5* (1 - 2 * float(training_metadata.episode) / training_metadata.num_episodes))'


# The explore rate decays from 0.6 to 0.1 linearly over the frame limit
# defined in the training_metadata and stays at 0.1 thereafter
class Atari_Explore_Rate:

    def get(self, training_metadata):
        return max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))

    def __str__(self):
        return 'max(0.1, 0.1 + 0.5*(1 - float(training_metadata.frame) / training_metadata.frame_limit))'


# The explore rate is fixed to 0.1 constantly
class Fixed_Explore_Rate:

    def get(self, training_metadata):
        return 0.1

    def __str__(self):
        return '0.1'