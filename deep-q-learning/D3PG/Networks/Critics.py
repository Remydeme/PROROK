import tensorflow.keras as k
import tensorflow as tf

class Critics:

    hidden_1 = 400
    hidden_2 = 300

    def __init__(self, env, lr=1e-3):
        input_dim = env.observation_space.shape
        action_dim = env.action_space.shape
        self.__lr = lr
        self.__initializer = k.initializers.he_uniform(seed=0)
        self.optimizer = k.optimizers.Adam(learning_rate=lr)
        self.valueNet = self.buildModel(input_dim=input_dim, action_dim=action_dim)
        self.targetValueNet = self.buildModel(input_dim=input_dim, action_dim=action_dim)
        self.targetValueNet.set_weights(self.valueNet.get_weights())



    def buildModel(self, input_dim, action_dim):

        model = k.Sequential()

        model.add(k.layers.Dense(self.hidden_1, input_dim=(input_dim[0] + action_dim[0]), activation='relu', name='hidden_1'))

        model.add(k.layers.Dense(self.hidden_2, activation='relu', name='hidden_action'))

        model.add(k.layers.Dense(1, name='output_layer'))

        return model

    def computeValue(self, state, actions):
        state_and_action = tf.concat([state, actions], axis=1)
        value = self.valueNet(state_and_action)
        return value

    def computeTargetValue(self, state, actions):
        state_and_action = tf.concat([state, actions], axis=1)
        value = self.targetValueNet(state_and_action)
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

    def computeLosses(self, value, target):
        critic_loss = k.losses.mean_squared_error(target, value)
        return critic_loss