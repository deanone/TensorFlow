import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
scaler = MinMaxScaler()
housing_data_plus_bias_scaled = scaler.fit_transform(housing_data_plus_bias)


n_epochs = 1000
learning_rate = 0.01
X = tf.constant(housing_data_plus_bias_scaled, dtype = tf.float32, name = "X")
y = tf.constant(housing.target.reshape(-1, 1), dtype = tf.float32, name = "y")
theta = tf.Variable(tf.compat.v1.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta")
y_pred = tf.matmul(X, theta, name = "predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name = "mse")
gradients = 2/m * tf.matmul(tf.transpose(X), error)
training_op = tf.compat.v1.assign(theta, theta - learning_rate * gradients)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
	sess.run(init)
	for epoch in range(n_epochs):
		if epoch % 100 == 0:
			print("Epoch: ", epoch, "MSE: ", mse.eval())
		sess.run(training_op)

	best_theta = theta.eval()