import tensorflow as tf

class DNN:

	# Hyperparameters for NN
	N_HIDDEN_1 = 10
	N_HIDDEN_2 = 10

	# Store layers weight & bias
	def __init__(self, num_input, num_output, LEARNING_RATE):
		self.num_input = num_input
		self.num_output = num_output
		self.LEARNING_RATE = LEARNING_RATE

		self.x = tf.placeholder(tf.float32, [None, self.num_input])
		with tf.name_scope('hidden1'):
			w1 = tf.Variable(tf.random_normal(shape=[self.num_input, N_HIDDEN_1],
				dtype=tf.float32), name='w1')
			h1 = tf.matmul(self.x, w1)
		with tf.name_scope('output'):
			w2 = tf.Variable(tf.random_normal(shape=[N_HIDDEN_1, self.num_output], 
				dtype=tf.float32), name='w2')
			self.Q = tf.matmul(h1, w2)

		self.weights = [w1, w2]

		# Loss
		self.targetQ = tf.placeholder(tf.float32, [None])
		self.targetActionMask = tf.placeholder(tf.float32, [None, self.num_output])
		# TODO: Optimize this
		q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask), 
			reduction_indices=[1])
		self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ)))

    	# Reguralization
    	# for w in [W1, W2, W3]:
     	# 	self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

    	# Training
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)

		self.init_op = tf.global_variables_initializer()