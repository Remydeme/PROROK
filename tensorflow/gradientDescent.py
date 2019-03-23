from sklearn.datasets import fetch_california_housing
import tensorflow as tf
import ssl
import numpy as np
from datetime import datetime

def prepareData():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    data_with_bias = np.c_[np.ones((m, 1)), housing.data]
    X = tf.constant(data_with_bias, dtype=tf.float32, name='X')
    y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
    return (X, y, m, n)


def getData():
    housing = fetch_california_housing()
    m, n = housing.data.shape
    x = np.c_[np.ones((m, 1)), housing.data]
    y = housing.target.reshape(-1,1)
    return (x, y, m, n)


def normalGradiant():
    """ Compute the normal gradiant
        theta = inv(t(X)* X) * t(X) *y
    """
    X, y, _, _ = prepareData()
    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
    with tf.Session():
        t = theta.eval()
        print("The value of theta is : {}".format(t))
    return t



def getBatch(X, y, size, index, column_size):
    X_batch = X[index:size, :]
    y_batch = y[index : size]
    return (X_batch , y_batch)



def getLogiDir():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)
    return logdir

def miniBatchGradientDescent(learning_rate=0.001, episodes=100, bacth_size=100):
    """
    This function compute the gradiant descent
    theta = theta - learning_rate * (2/m * (t(X) * (X * theta - y))
    :return:
    """
    X_dataset, y_dataset, m, n = getData()
    X = tf.placeholder(dtype=tf.float32,shape=(None, n + 1), name='X')
    y = tf.placeholder(dtype=tf.float32,shape=(None, 1), name='y')
    XT = tf.transpose(X)
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1, 1), name='theta')
    y_pred = tf.matmul(X, theta, name='prediction')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    gradient = 2 / m  * tf.matmul(XT, error)
    gradient_descent = tf.assign(theta, theta - learning_rate * gradient, name='operation_gradient')
    init = tf.global_variables_initializer()

    # saver
    saver = tf.train.Saver()
    # display the log
    logdir = getLogiDir()
    mse_summary = tf.summary.scalar("MSE", mse) # create mse_summary it will permit to eval the mse and store the result in a summary
    summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)
        for episode in range(episodes):
            index = 0
            while index < m:
                x_batch, y_batch = getBatch(X_dataset, y_dataset,m, index, column_size=n+1)
                if episode % 10:
                    summary_str = mse_summary.eval(feed_dict={X: x_batch, y: y_batch})
                    step = episode * bacth_size + index
                    summary_writer.add_summary(summary_str, step)
                sess.run(gradient_descent, feed_dict={X: x_batch, y: y_batch})
                index += bacth_size
                print(theta.eval())
    summary_writer.close()



if __name__ == "__main__":
    miniBatchGradientDescent()
