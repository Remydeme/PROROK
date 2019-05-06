import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import gym
import datetime

from collections import deque

class PGAgent():

    hidden_1_shape = 14
    hidden_2_shape = 7
    outputs_shape = 2
    episode_memory = []


    def __init__(self, input_shape, eta=0.001, gamma=0.95, baseline=1):
        self.__input_shape = input_shape
        self.__eta = eta
        self.__baseline = baseline
        self.__gamma = gamma
        self.__initializer = keras.initializers.he_uniform(seed=None)
        self.__model, self.__gradBuffer, self.__compute_loss, self.__optimizer = self.network()




    def network(self):
        """ This method is called to build a network """
        model = tf.keras.Sequential()

        # add a dense layer of hidden 1
        model.add(layers.Dense(self.hidden_1_shape, input_dim=self.__input_shape, activation='elu', kernel_initializer=self.__initializer))

        # add a dense layer hidden 2
        model.add(layers.Dense(self.hidden_2_shape, activation='elu', kernel_initializer=self.__initializer))

        # outputs
        # we have multiple ouput so we use softmax for one output we use the sigmoid function
        model.add(layers.Dense(self.outputs_shape, activation='softmax', kernel_initializer=self.__initializer))

        # build the model
        """
        This is to be used for subclassed models, which do not know at instantiation time what their inputs look like.
        This method only exists for users who want to call model.build() in a standalone way (as a substitute for calling 
        the model on real data to build it). It will never be called by the framework (and thus it will never throw 
        unexpected errors in an unrelated workflow).
        """
        model.build()
        
        """
        Keras optimizer 
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # sparse is used because our outputs don't have the one hot encoder outputs form.
        # we have integer value as ouputs
        compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        #model.compile(optimizer=optimizer, loss=compute_loss)




        # we need to keep all the gradient variable in a variable
        gradBuffer = model.trainable_variables
        for var, grad in enumerate(gradBuffer):
            gradBuffer[var] = grad * 0

        return model, gradBuffer, compute_loss, optimizer



    def takeAction(self, state):
        with tf.GradientTape() as tape:
            # forward pass
            # logits is a vector that contains the 2 probability
            logits = self.__model(state)
            a_dist = logits.numpy()
            # Choose random action with p = action dist
            # a = np.multinomial(np.log(a_dist[0]))
            a = np.random.choice(a_dist[0], p=a_dist[0])
            a = np.argmax(a_dist == a)
            loss = self.__compute_loss([a], logits)
        # compute the gradient of the loss function according trainable variable
        # exp : x = 3 | y = x * x
        # exp : tape.gradient(y, x) = > 2 x => 2 * 3 => 6
        # we need to store this gradient at each steps
        grads = tape.gradient(loss, self.__model.trainable_variables)
        return a, grads


    def store(self, episode_reward_gradient):
        self.episode_memory.append(episode_reward_gradient)

    def normalise(self, discount_reward):
        """
        :info function normalise the array of rewards get for each episode
        :param discount_reward:
        :return: numpy.ndArray
        """
        discount_std = discount_reward.std()
        discount_mean = discount_reward.mean()
        return [ (discounted_reward - discount_mean) / discount_std
                     for discounted_reward in discount_reward ]

    def discount_rate(self, rewards, gamma=0.95):
        """ Compute the discout rate for each step.
            Advantage = R(t) + gamma * r(t + 1) + gamma^2 * r(t + 2) + ... + gamma ^n * r(t + n)

            @ reward : is an array that contains all the reward of an episode
        """
        discounted_r = np.zeros_like(rewards)
        reward_sum = 0
        for t in reversed(range(0, rewards.size)):
            reward_sum = (reward_sum * gamma + rewards[t]) - self.__baseline
            discounted_r[t] = reward_sum
        return discounted_r

    def compute_gradient_and_discount_reward(self):
        """
            This method is called after an episode to compute the gradient
            using the technique of the policy gradient
            we compute the Advantage for each step and after we uodate the gradient
            by multiplying it by the advantage
        """
        self.episode_memory = np.array(self.episode_memory)
        self.episode_memory[:, 1] = self.discount_rate(self.episode_memory[:, 1])
        self.episode_memory[:, 1] = self.normalise(self.episode_memory[:, 1])

        for grads, r in self.episode_memory:
            for ix, grad in enumerate(grads):
                """
                    By doing this multiplication if an action was good the reward is positive and it will apply the 
                    gradient in order to encourage action like that in the future if the reward is negative it will 
                    avoid more to do this action in the future.  
                """
                self.__gradBuffer[ix] += grad * r


    def train(self):
        self.__optimizer.apply_gradients(zip(self.__gradBuffer, self.__model.trainable_variables))

    def resetGradientBuffer(self):
        for ix, grad in enumerate(self.__gradBuffer):
            self.__gradBuffer[ix] = grad * 0

    def resetMemory(self):
        """
        The episode memory buffer store the score of each step during the party
        After each game we need to clean it
        """
        # reset memory
        self.episode_memory = []

    def save_model(self, save_path):
        self.__model.save(save_path)



# ajouter Horizon de 1000 steps maximum
def trainOnGame(training_delay=100):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    env = gym.make('CartPole-v0')
    env.reset()
    scores = []
    model_filename = "PGAgentModel" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = PGAgent(input_shape=4)
    for e in range(3500):
        # reset the enviroment
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, 4])
            # make the choosen action
            a, grads = model.takeAction(state=s)
            s, r, done, info = env.step(a)
            ep_score += r
            if done:
                r -= 10  # small trick to make training faster
            model.store([grads, r])
        scores.append(ep_score)
        model.compute_gradient_and_discount_reward()
        model.resetMemory()
        if e % training_delay == 0:
            with train_summary_writer.as_default():
                model.train()
                model.resetGradientBuffer()

        if e % training_delay == 0:
            # compute the mean score on the 10 last episode
            print("Episode  {}  Score  {}".format(e, np.mean(scores[-training_delay:])))
            with train_summary_writer.as_default():
                tf.summary.scalar('mean_score',np.mean(scores[-training_delay:]), step=e)
        # save the model every 100 episode
        if e % training_delay == 0:
             model.save_model(model_filename)
    return model


def play():

    model = trainOnGame()
    env = gym.make('CartPole-v0')
    env.reset()
    for e in range(10000):
        # reset the enviroment
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, 4])
            a, grads = model.takeAction(state=s)
            s, r, done, _ = env.step(a)
            env.render()


if __name__ == "__main__":
    play()


