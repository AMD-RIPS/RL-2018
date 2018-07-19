import sys
sys.dont_write_bytecode = True

import tensorflow as tf


class Basic:

    def __init__(self):
        pass

    def get(self, episode, num_episodes):
        return max(0.01, 1 - float(episode)/((num_episodes+1)/2))


class AtariPaper:
	def __init__(self):
		pass

	def get(self, frames, frames_limit):
		return max(0.1, (1 - 0.9*(float(frames)/frames_limit)))


expl_dict = {
    'basic': Basic,
    'AtariPaper': AtariPaper
}
