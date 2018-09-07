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