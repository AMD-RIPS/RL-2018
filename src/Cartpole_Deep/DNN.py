import tensorflow as tf

class DNN:

	# Hyperparameters for NN
	N_HIDDEN_1 = 16
	N_HIDDEN_2 = 16

	# Store layers weight & bias
	def __init__(self, num_input, num_output, LEARNING_RATE):
		self.num_input = num_input
		self.num_output = num_output
		self.LEARNING_RATE = LEARNING_RATE

		self.x = tf.placeholder(tf.float32, [None, self.num_input])
		with tf.name_scope('hidden1'):
			w1 = tf.Variable(tf.random_normal(shape=[self.num_input, self.N_HIDDEN_1],
				dtype=tf.float32), name='w1')
			b1 = tf.Variable(tf.zeros(self.N_HIDDEN_1), name='b1')
			h1 = tf.nn.relu(tf.matmul(self.x, w1)+b1)
		with tf.name_scope('hidden2'):
			w2 = tf.Variable(tf.random_normal(shape=[self.N_HIDDEN_1, self.N_HIDDEN_2],
				dtype=tf.float32), name='w2')
			b2 = tf.Variable(tf.zeros(self.N_HIDDEN_2), name='b2')
			h2 = tf.nn.relu(tf.matmul(h1, w2)+b2)
		with tf.name_scope('output'):
			w3 = tf.Variable(tf.random_normal(shape=[self.N_HIDDEN_2, self.num_output], 
				dtype=tf.float32), name='w3')
			b3 = tf.Variable(tf.zeros(self.num_output), name='b3')
			self.Q = tf.matmul(h2, w3)+b3

		self.weights = [w1, b1, w2, b2, w3, b3]

		# Loss
		self.targetQ = tf.placeholder(tf.float32, [None])
		self.targetActionMask = tf.placeholder(tf.float32, [None, self.num_output])
		# TODO: Optimize this
		q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask),axis=1)
		self.loss = tf.reduce_sum(tf.square(tf.subtract(q_values, self.targetQ)))

    	# Reguralization
    	# for w in [W1, W2, W3]:
     	# 	self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

    	# Training
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		global_step = tf.Variable(0, name='global_step', trainable=False)
		self.train_op = optimizer.minimize(self.loss, global_step=global_step)

		self.init_op = tf.global_variables_initializer()