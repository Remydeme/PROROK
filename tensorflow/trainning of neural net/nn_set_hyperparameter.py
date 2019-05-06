import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



# dans ce fichier nous allons construire un réseau de neuronne à laide de tensorflow
# et utiliser différente technique afin de régler les hyperparamètre.


# input : picture of 28 * 28 pixels in black and white
# This neural net is a perceptron neural net with 3 hidden
# activation functions: leaky relu
# optimizer : adam
class NN():

    # Neural nets parameters
    input_size = 28 * 28
    hidden_1_size = 300
    hidden_2_size = 100
    hidden_3_size = 50
    ouput_size = 10

    # trainning parameters
    n_epochs = 400
    batch_size = 50



    def __init__(self, X, y):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size],name='X')
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, 1],name='X')
        self.model = self.build(X, y)




    def load_data(self):
        mnist = input_data.read_data_sets('/tmp/data/')
        self.n_batch_per_epochs = mnist.train.num_examples
        self.X_train = mnist.train.images
        self.X_test = mnist.test.images
        self.y_train = mnist.train.labels.astype("int")
        self.y_test = mnist.test.labels.astype("int")


    def build(self, inputs, target):
        with tf.name_scope('layers'):
            he_init = tf.contrib.layers.variance_scaling_initializer()
            X = tf.layers.dense(inputs, self.input_size, activation=tf.nn.leaky_relu, kernel_initializer=he_init, name='X')
            h1 = tf.layers.dense(X, self.hidden_1_size, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
                                 name='hidden_1')
            h2 = tf.layers.dense(h1, self.hidden_2_size, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
                                 name='hidden_2')
            h3 = tf.layers.dense(h2, self.hidden_3_size, activation=tf.nn.leaky_relu, kernel_initializer=he_init,
                                 name='hidden_3')
            logits = tf.layers.dense(h3, self.hidden_3_size, activation=tf.nn.leaky_relu, name='outputs')

            # add the cost function
            # we use cross entropy function with logits. It's will compute the error for each category before the value Z pass by the activation function
            # The entropy of each category is computed and store in a tensor. The mean of entropy is then caculated using the tf.readuce_mean function
        with tf.name_scope('loss'):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, target)
            loss = tf.reduce_mean(xentropy, name='loss')  # compute the mean error of our neural net

            # Now we have to implement the gradient descent to improve our weights.
            learning_rate = 0.01
        with tf.name_scope('gradient'):
            optimizer = tf.train.AdamOptimizer()
            training_op = optimizer.minimize(loss)  # minimize the loss function

            # measure the accuracy
        with tf.name_scope('eval'):
            correct = tf.nn.in_top_k(logits, target, 1)
            accuracy = tf.reduce_mean(tf.cast(logits, tf.float32))
        return training_op


    # scope : name of the scope use for all the layer
    # kernel_initializer : We init the weight using he technic.
    def fit(self, inputs, target):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            init.run()
            for epoch in range(self.n_epochs):
                for iteration in range(n_batches_per_epochs):
                    X_batch, y_batch = mnist.train.next_batch(self.batch_size)

            sess.run(self.model)

        return self









if __name__ == "__main__":
    nn = NN()
