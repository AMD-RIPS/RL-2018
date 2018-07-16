import sys
sys.dont_write_bytecode = True

import tensorflow as tf

class Basic:
	def __init__(self):
		pass

	def get(self):
		return 0.1

expl_dict = {
	'basic': Basic
}
