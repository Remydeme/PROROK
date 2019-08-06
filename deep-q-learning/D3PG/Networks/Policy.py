import tensorflow as tf
import tensorflow.keras as k

class Model(k.Model):

    hidden_1_size = 400
    hidden_2_size = 300

    def __init__(self, action_dim, action_max):
        super().__init__()
        self.action_max = action_max
        self.__initializer = k.initializers.he_uniform(seed=0)
        self.h1 = k.layers.Dense(self.hidden_1_size, activation='relu', name='hidden_1')
        self.h2 = k.layers.Dense(self.hidden_2_size, activation='relu', name='hidden_2')
        self.outputs = k.layers.Dense(action_dim, activation='tanh', name='actions_ouput')



    def call(self, inputs):
        x = self.h1(inputs)
        x = self.h2(x)
        x = (self.action_max * self.outputs(x))
        return x




class Policy:



    def __init__(self, action_dim, action_high, lr=1e-3):
        self.action_max = action_high[0]
        self.action_dim = action_dim
        self.policyNet = Model(action_dim=action_dim, action_max=self.action_max)
        self.policyTargetNet = Model(action_dim=action_dim, action_max=self.action_max)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.policyTargetNet.set_weights(self.policyNet.get_weights())

    def get_action(self, state):
        actions = self.policyNet(state)
        return actions[0]

    def get_target_action(self, state):
        actions = self.policyTargetNet(state)
        return actions

    def evaluate_state(self, state):
        actions = self.policyNet(state)
        return actions




    def softCopy(self, tau=0.005):
        target_pars = self.policyTargetNet.get_weights()
        value_pars = self.policyNet.get_weights()
        index = 0
        for target_par, value_par in zip(target_pars, value_pars):
            target_par = target_par * (1 - tau) + value_par * tau
            target_pars[index] = target_par
            index += 1
        self.policyTargetNet.set_weights(target_pars)

