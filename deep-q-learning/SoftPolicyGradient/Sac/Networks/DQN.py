
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as kl
import numpy as np
from datetime import datetime



class DQN():

    _hidden_1 = 128
    _hidden_2 = 64

    def __init__(self, input_size,action_size, lr=1e-3):
        self.__input_size = input_size
        self.__action_size = action_size
        self.__lr = lr
        self.__initializer = tf.keras.initializers.he_uniform(seed=7)
        self.__hiden_3_init = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)
        self.__qNet, self.__qOptimizer = self._build_net()
        self.__targetNet,  self.__targetOptimizer = self._build_net()




    def _build_net(self):

        model = tf.keras.Sequential()

        model.add(layers.Dense(self._hidden_1, input_dim=(self.__input_size + self.__action_size), activation='elu', kernel_initializer=self.__initializer))

        model.add(layers.Dense(self._hidden_2, activation='elu', kernel_initializer=self.__initializer))

        # initialize the last layers with uniform intialization std = 0.3

        model.add(layers.Dense(1, activation='linear', kernel_initializer=self.__hiden_3_init))

        model.build()

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__lr)


        model.compile(optimizer=optimizer, loss='mse')

        return model, optimizer



    def train(self, target, state_and_action):
        """
        This function take a tensor of target value computed earlier and a tensor of value and compute 1/size * sum(target - value)^2
        The we train our network my minimizing this error
        :param target:
        :param value:
        :return:
        """
        self.__qNet.fit(state_and_action, target)
        """with tf.GradientTape() as tape:
            mse = tf.keras.losses.mean_squared_error(target, value)
        print("Mse error {}".format(mse))
        for error in mse:
            print("error : {}".format(error))
            gradient = tape.gradient(error, self.__qNet.trainable_variables)
            print("gradient : {}".format(gradient))
            self.__qOptimizer.apply_gradients(zip(gradient, self.__qNet.trainable_variables))
        return mse"""

    def train_target(self, target, state_and_action):
        """
        :param target:
        :param value:
        :return:
        """
        self.__targetNet.fit(state_and_action, target)
        """mse = 0
        with tf.GradientTape() as tape:
            mse = tf.keras.losses.mean_squared_error(target, value)
        gradient = tape.gradient(mse, self.__targetNet.trainable_variables)
        self.__targetOptimizer.apply_gradients(zip(gradient, self.__targetNet.trainable_variables))
        return mse"""

    def computeQ(self, observation):
        """
        :param observation:
        :return Qvalue:
        """
        action = self.__qNet(observation)
        return action

    def updateTargetNetwork(self, soft_tau=0.8):
        """
        Copy the wieghts of the Qnet in the target network
        """
        pars_behavior = self.__qNet.get_weights()  # these have form [W1, b1, W2, b2, ..], Wi =
        pars_target = self.__targetNet.get_weights()  # bi = biases in layer i

        ctr = 0
        for par_behavior, par_target in zip(pars_behavior, pars_target):
            par_target = par_target * (1 - self.tau) + par_behavior * self.tau
            pars_target[ctr] = par_target
            ctr += 1

        self.__targetNet.set_weights(pars_target)


    def computeTaget(self, observation):
        """
        :param observation:
        :return:
        """
        action = self.__targetNet(observation)
        return action

    def save(self):
        model_filename = "SAC_DQN_Model" + datetime.now().strftime("%Y%m%d-%H%M%S")
        target_model_filename = "SAC_DQN_Target_Model" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.__qNet.save(model_filename)
        self.__targetNet.save(target_model_filename)




if __name__ == "__main__":

    dqn = DQN(input_size=2,action_size=2)
    print(dqn.computeQ(np.array([[2, 3, 1.5, 0.5]])))