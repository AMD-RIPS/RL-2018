import sys
sys.dont_write_bytecode = True

import tensorflow as tf

class Basic:
	def __init__(self):
		pass

	def get(self):
		return 0.001

lrng_dict = {
	'basic': Basic
}
