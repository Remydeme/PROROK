


import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as kl
import numpy as np
from datetime import datetime

class ValueNetworks():

    hidden_1 = 128
    hidden_2 = 64


    def __init__(self, input_size,  lr=1e-3):
        self.__lr = lr
        self.__input_size = input_size
        self.__initializer = tf.keras.initializers.he_uniform(seed=7)
        self.__output_init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)
        self.__valueNet, self.__valueOptimizer = self._build_network()
        self.__targetNet,self.__targetNetOptimizer = self._build_network()


    def _build_network(self):

        model = tf.keras.Sequential()

        model.add(layers.Dense(self.hidden_1, input_dim=self.__input_size, activation='elu', kernel_initializer=self.__initializer))

        model.add(layers.Dense(self.hidden_2,  activation='elu', kernel_initializer=self.__initializer))

        model.add(layers.Dense(1, activation='linear', kernel_initializer=self.__output_init))

        model.build()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__lr)

        model.compile(optimizer=optimizer, loss='mse')

        return model, optimizer


    def computeValue(self, observation):
        """
        compute the value using observation as input
        :param observation:
        :return value:
        """
        value = self.__valueNet(observation)
        return value

    def computeTargetValue(self, observation):
        """
        compute the taget value using observation as input
        :param observation:
        :return value:
        """
        value = self.__targetNet(observation)
        return value

    def updateTarget(self, tau=0.2):
        """
        Copy the weights of the value network into the target network
        :return:
        """
        pars_behavior = self.__valueNet.get_weights()  # these have form [W1, b1, W2, b2, ..], Wi =
        pars_target = self.__targetNet.get_weights()  # bi = biases in layer i

        ctr = 0
        for par_behavior, par_target in zip(pars_behavior, pars_target):
            par_target = par_target *  (1 - tau) + tau * par_behavior
            pars_target[ctr] = par_target
            ctr += 1

        self.__targetNet.set_weights(pars_target)


    def train(self, observations, value):
        """
       mse = tf.keras.losses.mean_squared_error(target, value)
        with tf.GradientTape() as tape:
            gradient = tape.gradient(mse, self.__valueNet.trainable_variables)
            self.__valueOptimizer.apply_gradients(zip(gradient, self.__valueNet.trainable_variables))
        return mse
        """
        self.__valueNet.fit(observations, value)



    def save(self):
        model_filename = "SACValueModel" + datetime.now().strftime("%Y%m%d-%H%M%S")
        target_model_filename = "SACTragetValueModel" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.__valueNet.save(model_filename)
        self.__targetNet.save(target_model_filename)

