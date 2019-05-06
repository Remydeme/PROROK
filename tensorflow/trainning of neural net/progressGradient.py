import tensorflow as tf

import numpy as np



""" 

class Agent():
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1

    learning_rate = 0.01

    def __init__(self):
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.model, self.init , self.saver = self.build()




    def build(self):
        

        X = tf.placeholder(dtype=tf.float32, shape=[None, self.n_inputs], name='X')

        hidden1 = tf.layers.dense(X, self.n_hidden,  activation=tf.nn.elu, kernel_initializer=self.initializer, name='hidden1')

        logits = tf.layers.dense(hidden1, self.n_outputs,  kernel_initializer=self.initializer, name='logits')

        outputs = tf.nn.sigmoid(logits, name='outputs')

        # outputs is a tensor with one value close to 1 if the command is to go right and close to 0 the action is left
        #we concat the output with 1 - output we get the propability for action to be right
        p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])

        # choose a randomly the next action
        action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

        y = 1 - tf.float(action)

        # compute the error using cross entropy
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits, name='cross_entropy')

        # Descent gradient using with adam optimizer
        # Use inertie moment to accelerate the descent gradient and we have better precision
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # return tensor of gradient and variables
        grad_and_vars = optimizer.compute_gradients(cross_entropy)

        # store the gradient in a array

        gradients = [grad for grad , variable in grad_and_vars]@

        gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grad_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())

            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((grad, variable))
        training_op = optimizer.apply_gradients(grads_and_vars_feed)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        return (training_op, init, saver)













"""

