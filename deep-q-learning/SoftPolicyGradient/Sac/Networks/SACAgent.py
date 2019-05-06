

from SoftPolicyGradient.Sac.Networks.DQN import DQN
from SoftPolicyGradient.Sac.Networks.ValueNetwork import ValueNetworks
from SoftPolicyGradient.Sac.Networks.PolicyNetwork import PolicyNetwork
from SoftPolicyGradient.Sac.BufferReplay import ReplayBuffer

import tensorflow as tf
import numpy as np
from datetime import datetime
import gym
import roboschool

class SACAgent():

    replayBufferSize = 100000

    def __init__(self, action_size, input_size, lr=1e-3, gamma=0.95):
        self.__gamma = gamma
        self.Qnets = DQN(input_size=input_size, action_size=action_size,lr=lr)
        self.ValueNet = ValueNetworks(input_size=input_size)
        self.policyNet = PolicyNetwork(input_size=input_size, action_shape=action_size)
        self.replayBuffer = ReplayBuffer(obs_dim=input_size,act_dim=action_size, size=self.replayBufferSize)


    def get_action(self, observation):
        a = self.policyNet.get_action(observation=observation)
        return a


    def update(self, batch_size, soft_tau=1e-2):

        with tf.GradientTape() as tape:

            state,next_state, actions, reward,done = self.replayBuffer.sample_batch(batch_size=batch_size)

            # Keras want input of type tf.float64 by default in depend on the processor
            state = tf.convert_to_tensor(state, dtype=tf.float64)
            next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
            actions = tf.convert_to_tensor(actions, dtype=tf.float64)
            reward = tf.convert_to_tensor(reward, dtype=tf.float64)
            done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)


            new_action, log_prob, epsilon, mean, log_std = self.policyNet.evaluate_action(observation=state)

            # Training Q Function

            target_value = self.ValueNet.computeTargetValue(observation=next_state)

            target_q_value = reward + (1 - done) * tf.cast(target_value, dtype=tf.float64) * self.__gamma

            state_and_action = tf.concat([state, actions], axis=1)


            self.Qnets.train(target_q_value, state_and_action)

            self.Qnets.train_target(target_q_value, state_and_action)

            # Training Value Function
            state_and_new_action = tf.concat([state, actions], axis=1)

            predicted_new_q_value = self.Qnets.computeQ(observation=state_and_new_action)

            predicted_new_q_target_value = self.Qnets.computeTaget(observation=state_and_new_action)

            predicted_new_q_value = tf.math.minimum(predicted_new_q_value, predicted_new_q_target_value)

            # loss for value network

            target_value_func = tf.cast(predicted_new_q_value, dtype=tf.float64) - log_prob

            self.ValueNet.train(observations=state, value=target_value_func)

            # train policy network

            #self.policyNet.train(observation=observations, log=log_prob, target=predicted_new_q_value)
            predicted_new_q_value = tf.cast(predicted_new_q_value, dtype=tf.float64)
            value_net = self.ValueNet.computeValue(observation=state)
            log_prob_target = predicted_new_q_value - tf.cast(value_net, dtype=tf.float64)
            policy_loss = (log_prob * (log_prob - log_prob_target))
            policy_loss = tf.math.reduce_sum(policy_loss)
            policy_loss += (1e-3 * (tf.math.reduce_sum(tf.math.square(mean))
                                   + tf.math.reduce_sum(tf.math.square(log_std))))
            policy_loss = policy_loss
            print("policy loss : {}".format(policy_loss))
            gradient = tape.gradient(policy_loss, self.policyNet.model.trainable_variables)
            self.policyNet.optimizer.apply_gradients(zip(gradient, self.policyNet.model.trainable_variables))


            # soft copy of the parameters of the value network in the qNetwork
            self.ValueNet.updateTarget(tau=soft_tau)

        def save_model(self):
            self.ValueNet.save()
            self.Qnets.save()
            self.policyNet.save()


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action
############## Game LOOP ###############



def play():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    env = NormalizedActions(gym.make('Pendulum-v0'))
    scores = []
    agent = SACAgent(action_size=env.action_space.shape[0], input_size=env.observation_space.shape[0])
    training_delay = 1
    batch_size = 32
    for e in range(10000):
        s = env.reset()
        ep_score = 0
        done = False
        for step in range(500):
            # make the chosen action
            s = s.reshape([1, env.observation_space.shape[0]])
            a = agent.get_action(observation=s)
            next_state, r, done, _ = env.step(a)
            ep_score += r
            agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)
            s = next_state
            if agent.replayBuffer.size > batch_size:
                agent.update(batch_size=batch_size)
        scores.append(ep_score)
        if e % training_delay == 0:
            # compute the mean score on the 10 last episode
            print("Episode  {}  Score  {}".format(e, np.mean(scores[-training_delay:])))
            with train_summary_writer.as_default():
                tf.summary.scalar('mean_score', np.mean(scores[-training_delay:]), step=e)
        # save the model every 100 episode
        if e % training_delay == 0:
            pass
           #agent.save_model()
    return agent


if __name__ == "__main__":
    model = play()
    env = NormalizedActions(gym.make('Pendulum-v0'))
    env.reset()
    for e in range(100):
        # reset the enviroment
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, env.observation_space.shape[0]])
            a = model.get_action(observation=s)
            s, r, done, _ = env.step(a)
            env.render()
        """
            model = play()
            env = gym.make('RoboschoolHumanoid-v1')
            env.reset()
            for e in range(10000):
                # reset the enviroment
                s = env.reset()
                ep_memory = []
                ep_score = 0
                done = False
                while not done:
                    s = s.reshape([1, 3])
                    a = model.get_action(observation=s)
                    s, r, done, _ = env.step(a)
                    env.render()
                    
            env = gym.make('Pendulum-v0')
            agent = SACAgent(action_size=env.action_space.shape[0], input_size=env.observation_space.shape[0])
            s = env.reset()
            s = s.reshape([1, 3])
            a = agent.get_action(observation=s)
            print("Final action keep {}".format(a))
            
            """

















