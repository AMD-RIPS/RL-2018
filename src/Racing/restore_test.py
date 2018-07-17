import tensorflow as tf

# Let's load a previously saved meta graph in the default graph
# This function returns a Saver
saver = tf.train.import_meta_graph('data-all.chkp.meta')

# We can now access the default graph where all our metadata has been loaded
graph = tf.get_default_graph()


# Finally we can retrieve tensors, operations, collections, etc.
train_op = graph.get_operation_by_name('loss/train_op')

with tf.Session() as sess:
	saver.restore(sess, 'data-all.chkp.data-00000-of-00001')
	print(sess.run(global_step_tensor)) # returns 1000