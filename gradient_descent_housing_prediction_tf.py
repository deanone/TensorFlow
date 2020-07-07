from datetime import datetime
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


tf.compat.v1.disable_eager_execution()


def fetch_batch(X, y, batch_index, batch_size):
	X_batch = X[(batch_index * batch_size) : ((batch_index + 1) * batch_size), :]
	y_batch = y[(batch_index * batch_size) : ((batch_index + 1) * batch_size)]
	return X_batch, y_batch


def main():

	assert not tf.executing_eagerly()

	now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
	root_logdir = 'tf_logs'
	logdir = '{}/run-{}/'.format(root_logdir, now)

	housing = fetch_california_housing()
	m, n = housing.data.shape
	housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
	scaler = MinMaxScaler()
	housing_data_plus_bias_scaled = scaler.fit_transform(housing_data_plus_bias)
	X = housing_data_plus_bias_scaled
	print(X.dtype)
	y = housing.target.reshape(-1, 1)

	n_epochs = 100
	learning_rate = 0.01
	batch_size =  100
	n_batches = int(np.ceil(m / batch_size))
	grad = 'optimizer'

	X_batch = tf.compat.v1.placeholder(tf.float32, shape = (None, n + 1), name = 'X_batch')
	y_batch = tf.compat.v1.placeholder(tf.float32, shape = (None, 1), name = 'y_batch')

	theta = tf.Variable(tf.compat.v1.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta")
	y_pred = tf.matmul(X_batch, theta, name = "predictions")
	error = y_pred - y_batch
	mse = tf.reduce_mean(tf.square(error), name = "mse")

	if grad == 'autodiff':
		gradients = tf.gradients(mse, [theta])[0]
		training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)
	elif grad == 'optimizer':
		optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate)
		training_op = optimizer.minimize(mse)
	else:
		gradients = 2/m * tf.matmul(tf.transpose(X_batch), error)
		training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

	init = tf.compat.v1.global_variables_initializer()

	mse_summary = tf.compat.v1.summary.scalar('MSE', mse)
	file_writer = tf.compat.v1.summary.FileWriter(logdir, tf.compat.v1.get_default_graph())

	# for saving the model
	#saver = tf.compat.v1.train.Saver()

	with tf.compat.v1.Session() as sess:
		sess.run(init)
		for epoch in range(n_epochs):
			#if epoch % 100 == 0:
				#print("Epoch: ", epoch, "MSE: ", mse.eval())
				#save_path = saver.save(sess, 'model.ckpt')
			for batch_index in range(n_batches):
				X_batch_fetched, y_batch_fetched = fetch_batch(X, y, batch_index, batch_size)
				print('Epoch: {}, Batch: {}, MSE: {}'.format(epoch, batch_index, mse.eval(feed_dict = {X_batch: X_batch_fetched, y_batch: y_batch_fetched})))

				if batch_index % 10 == 0:
					summary_str = mse_summary.eval(feed_dict = {X_batch: X_batch_fetched, y_batch: y_batch_fetched})
					step = epoch * n_batches + batch_index
					file_writer.add_summary(summary_str, step)

				sess.run(training_op, feed_dict = {X_batch: X_batch_fetched, y_batch: y_batch_fetched})

		best_theta = theta.eval()
		#save_path = saver.save(sess, 'final_model.ckpt')


if __name__ == '__main__':
	main()