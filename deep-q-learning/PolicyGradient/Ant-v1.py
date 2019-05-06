import tensorflow as tf
import gym
import roboschool

import roboschool
import gym


class Agent():
    replay = []
    inputs_shape = 14
    hidden_1_shape = 28
    hidden_2_shape = 28
    hidden_3_shape = 16
    outputs_shape = 16
    hidden_activation = tf.nn.elu

    def __init__():
        pass


env = gym.make('RoboschoolAnt-v1')
for i in range(1000):
    env.reset()
    while True:
        action = env.action_space.sample()
        print("Action" + str(action))
        print(action.shape)
        obs, reward, done, info = env.step(action)
        print(obs.shape)
        print(reward)
        env.render()
        if done:
            break

