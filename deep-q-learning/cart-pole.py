# -*- coding: utf-8 -*-
"""deep-q-learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jeN8am0akHaXYMbnwq_o_tjnj7VIA_F4
"""

# This project is a step of the PROROK project. We use openAI to learn How to make deep reinforcement learning.



import gym
import random
import numpy as np
from collections import deque
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
from keras import Model, Sequential
from keras.layers import Dense

from keras.optimizers import Adam
"""# **1 - Set environment variables**"""

# learning rate
lr = 1e-3
env = gym.make('CartPole-v0')
env.reset()
steps = 500
# score requirement value
score_minimum = 50
#number_of game
initial_game = 1000

"""# **2 - Play a with random action **"""

def playRandomly():
  for episode in range(100):
    env.reset()
    for step in range(steps):
      env.render() # we will see the action
      randomAction = env.action_space.sample() # thi function return a random action bae on the environment in the game
      print(randomAction, flush=True)
      observation, reward, done, info = env.step(randomAction) # play the action
      break
      if done == True:
        break

# let's try our new function

class DeepQAgent():

    episode = 1000

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.3
        self.lr = 1e-3
        self.gamma = 0.9
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)


    def reset(self):
        self.memory = []
        self.model = self.build_model()




    def build_model(self):

        model = Sequential()

        model.add(Dense(40, input_dim=self.state_size,kernel_initializer='uniform',  activation='relu'))

        model.add(Dense(40, kernel_initializer='uniform', activation='relu'))

        model.add(Dense(40, kernel_initializer='uniform', activation='relu'))

        model.add(Dense(40, kernel_initializer='uniform', activation='linear'))

        model.compile(loss='mse',optimizer=Adam(lr=self.lr))

        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        if act_values[0][0] > act_values[0][1]:
            return 0
        else:
            return 1


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))



    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




def trainAgent():
    env = gym.make('CartPole-v0')
    agent = DeepQAgent(state_size=4, action_size=2)
    # Iterate the game
    for e in range(agent.episode):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(3000):
            # turn this on if you want to render
            # env.render()
            # Decide action
            action = agent.act(state)

           # print(action, flush=True)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, agent.episode, time_t))
                break
        # train the agent with the experience of the episode
        agent.replay(32)
    return agent

if __name__ == "__main__":
    agent = trainAgent()
    for episode in range(100):
        state = env.reset()
        env.reset()
        for step in range(steps):
            env.render()  # we will see the action
            state = np.reshape(state, [1, 4])
            observation, reward, done, info = env.step(agent.act(state))  # play the action
            state = observation
            if done == True:
                break


