import tensorflow as tf
import tensorflow.keras as k
import numpy as np

class Model(k.Model):

    hidden_1_size = 256
    hidden_2_size = 256

    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.__initializer = k.initializers.he_uniform(seed=42)
        self.h1 = k.layers.Dense(self.hidden_1_size, activation='elu', kernel_initializer=self.__initializer)
        self.h2 = k.layers.Dense(self.hidden_2_size, activation='elu', kernel_initializer=self.__initializer)
        self.outputs = k.layers.Dense(action_dim, activation='tanh', kernel_initializer=self.__initializer)




    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        x = self.outputs(x)
        return x




class Policy():



    def __init__(self, input_dim, action_dim, action_low, action_high, lr=1e-4):
        self.action_low = action_low
        self.action_high = action_high
        self.action_dim = action_dim
        self.policyNet = Model(input_dim=input_dim, action_dim=action_dim)
        self.policyTargetNet = Model(input_dim=input_dim, action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.policyTargetNet.set_weights(self.policyNet.get_weights())


    def get_action(self, state):
        actions = self.policyNet(state)
        return actions[0]

    def evaluate_state(self, state):
        actions = self.policyNet(state)
        return actions


    def get_target_action(self, state):
        actions = self.policyTargetNet(state)
        return actions

    def softCopy(self, tau=1e-2):
        target_pars = self.policyTargetNet.get_weights()
        value_pars = self.policyNet.get_weights()
        index = 0
        for target_par, value_par in zip(target_pars, value_pars):
            target_par = target_par * (1 - tau) + value_par * tau
            target_pars[index] = target_par
            index += 1
        self.policyTargetNet.set_weights(target_pars)

