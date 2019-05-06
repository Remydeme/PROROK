
import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
from datetime import datetime
import math

class Model(tf.keras.Model):

    hidden_1_size = 128
    hidden_2_size = 64

    def __init__(self, input_size, action_size, log_std_min=-2, log_std_max=2):
        super().__init__('mlp_policy')
        # no tf.get_variable(), just simple Keras API
        self.__log_std_min = log_std_min
        self.__log_std_max = log_std_max
        self.__initializer = tf.keras.initializers.he_uniform(seed=7)
        self.hidden1 = layers.Dense(self.hidden_1_size, activation='elu', kernel_initializer=self.__initializer)
        self.hidden2 = layers.Dense(self.hidden_2_size, activation='elu', kernel_initializer=self.__initializer)

        # mean layers
        init_mean = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)
        self.mean = layers.Dense(action_size, kernel_initializer=init_mean, name='mean')


        # std layer
        init_std = tf.keras.initializers.RandomUniform(minval=-3e-3, maxval=3e-3, seed=None)

        self.std = layers.Dense(action_size, kernel_initializer=init_std,  name='std')

    def call(self, inputs):
        # inputs is a numpy array, convert to Tensor
        # separate hidden layers from the same input tensor
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        return self.mean(x), tf.clip_by_value(self.std(x), -20, 2)




class PolicyNetwork():


    def __init__(self, input_size, action_shape, lr=1e-3, gamma=0.99, entropy=0.5):
        self.__input_size = input_size
        self._action_shape = action_shape
        self.__lr = lr
        self.__gamma = gamma
        self.__entropy = entropy
        self.__initializer = tf.keras.initializers.he_uniform(seed=7)
        self.model = Model(input_size=self.__input_size, action_size=self._action_shape)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.__lr)



    def evaluate_action(self, observation, epsilon=1e-6):
        mean, std = self.model(observation)
        std = tf.math.exp(std)
        z = np.random.normal(mean[0], std[0], 1)
        action = tf.math.tanh(z)
        # log_prob = tf.math.log(self.normal(x=(mean + std * z), sigma=sigma, mu=mean)) - tf.math.log(1 - tf.square(action) + epsilon)
        p1 = - tf.math.square(action - mean) / 2 * tf.square(std)
        p2 = - tf.math.log(tf.math.sqrt(2 * math.pi * tf.square(std)))
        log_prob = p1 + p2
        log_prob = log_prob - tf.math.log(1 - tf.math.square(action) + epsilon)
        log_prob = tf.math.reduce_sum(log_prob)
        #print("mean {} , std {}, z {} action {}, log_prob {}".format(mean, std, z, action, log_prob))
        return action, log_prob, z, mean, std



    def get_action(self, observation):
        mean, std = self.model(observation)
        std = tf.math.exp(std)
        z = np.random.normal(mean[0], std[0], 1)
        action = tf.math.tanh(z)
        # print("mean {} , std {}, z {} action {}".format(mean, std, z, action))
        return action[0]

    def train(self, observation, log, target):
        target = tf.cast(target, dtype=tf.float64)
        log_loss = tf.reduce_mean(log - target)
        self.model.fit(x=observation, y=[log_loss, log_loss])
        # print("log loss {}".format(log_loss))

    def save(self):
        pass
    # model_filename = "SAC_Policy_Model" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # self.__model.save(model_filename)





if __name__ == "__main__":
    print(tf.constant([123, 3, 2,2], dtype=tf.float32))
