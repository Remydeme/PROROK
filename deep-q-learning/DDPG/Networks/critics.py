import tensorflow as tf
import tensorflow.keras as k




class Critics():

    hidden_1_size = 256
    hidden_2_size = 256

    def __init__(self, env, lr=1e-3):
        input_dim = env.observation_space.shape
        action_dim = env.action_space.shape
        self.__lr = lr
        self.__initializer = k.initializers.he_uniform(seed=7)
        self.valueNet = self.buildModel(input_dim=input_dim, action_dim=action_dim)
        self.targetValueNet = self.buildModel(input_dim=input_dim, action_dim=action_dim)
        self.valueNet.set_weights(self.targetValueNet.get_weights())

    def buildModel(self, input_dim, action_dim):


        state_input = k.Input(shape=input_dim)

        state_h1 = k.layers.Dense(self.hidden_1_size, activation='elu', kernel_initializer=self.__initializer)(state_input)

        state_h2 = k.layers.Dense(self.hidden_2_size, activation='elu', kernel_initializer=self.__initializer)(state_h1)

        action_input = k.layers.Input(shape=action_dim)

        action_h1 = k.layers.Dense(self.hidden_2_size, activation='elu', kernel_initializer=self.__initializer)(action_input)

        merged = k.layers.Add()([state_h2, action_h1])

        merge_h1 = k.layers.Dense(self.hidden_2_size, activation='elu', kernel_initializer=self.__initializer)(merged)

        output = k.layers.Dense(1)(merge_h1)

        model = k.Model(inputs=[state_input, action_input], outputs=output)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__lr)

        model.compile(optimizer=optimizer, loss='mse')

        return model






    def computeValue(self, state, actions):
        value = self.valueNet([state, actions])
        return value

    def computeTargetValue(self, state, actions):
        value = self.targetValueNet([state, actions])
        return value


    def softCopy(self, tau=1e-2):
        target_pars = self.targetValueNet.get_weights()
        value_pars = self.valueNet.get_weights()
        index = 0
        for target_par, value_par in zip(target_pars, value_pars):
            target_par = target_par * (1 - tau) + value_par * tau
            target_pars[index] = target_par
            index += 1
        self.targetValueNet.set_weights(target_pars)

    def train(self, state, actions, target):
        self.valueNet.fit([state, actions], target)