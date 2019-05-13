import tensorflow.keras as k
import tensorflow as tf

class Critics():
    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, env, lr=1e-3):
        input_dim = env.observation_space.shape
        action_dim = env.action_space.shape
        self.__lr = lr
        self.__initializer = k.initializers.he_uniform(seed=0)
        self.valueNet = self.buildModel(input_dim=input_dim[0], action_dim=action_dim[0])
        self.targetValueNet = self.buildModel(input_dim=input_dim[0], action_dim=action_dim[0])
        self.valueNet.set_weights(self.targetValueNet.get_weights())

    def buildModel(self, input_dim, action_dim):

        model = k.Sequential()

        model.add(k.layers.Dense(self.hidden_1, input_dim=(input_dim + action_dim), activation='relu', name='hidden_1'))

        model.add(k.layers.Dense(self.hidden_2, activation='relu', name='hidden_action'))

        model.add(k.layers.Dense(1, activation='linear', name='output_layer'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.__lr)

        model.compile(optimizer=optimizer, loss='mse')

        return model



    def computeValue(self, state, actions):
        state_and_action = tf.concat([state, actions], axis=1)
        value = self.valueNet(state_and_action)
        return value

    def computeTargetValue(self, state, actions):
        state_and_action = tf.concat([state, actions], axis=1)
        value = self.valueNet(state_and_action)
        return value

    def softCopy(self, tau=0.005):
        target_pars = self.targetValueNet.get_weights()
        value_pars = self.valueNet.get_weights()
        index = 0
        for target_par, value_par in zip(target_pars, value_pars):
            target_par = target_par * (1 - tau) + value_par * tau
            target_pars[index] = target_par
            index += 1
        self.targetValueNet.set_weights(target_pars)

    def train(self, state, actions, target):
        state_and_action = tf.concat([state, actions], axis=1)
        critic_loss = self.valueNet.train_on_batch(state_and_action, target)
        return critic_loss