from DDPG.Networks.Critics import Critics
from DDPG.Networks.Policy import Policy
from DDPG.replayBuffer import ReplayBuffer
import roboschool
import tensorflow as tf
import numpy as np

from DDPG.OUNoise import OUNoise

from datetime import datetime

from DDPG.normalizedAction import NormalizedActions

import gym

class DDPGAgent():

    def __init__(self, env, gamma=0.99, game_name='MountainCarContinuous-v0'):
        self.env = env
        self.game_name = game_name
        action_dim = env.action_space.shape
        input_dim = env.observation_space.shape
        self.policy = Policy(input_dim=input_dim[0], action_dim=action_dim[0], action_low=env.action_space.low,
                             action_high=env.action_space.high)
        self.critic = Critics(env=env)
        self.replayBuffer = ReplayBuffer(obs_dim=input_dim[0], act_dim=action_dim[0], size=1000000)
        self.__gamma = gamma
        self.noise = OUNoise(env.action_space)
        self.policy_losses = []
        self.critic_losses = []

    def get_action(self, state, step):
        actions = self.policy.get_action(state=state)
        return actions.numpy()

    def take_action(self, state):
        actions = self.policy.get_action(state=state)
        return actions

    def updateNetworks(self, batch_size=32, iterations=1000):

       # for step in range(iterations):
        state, next_state, actions, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

        # Keras want input of type tf.float64 by default in depend on the processor
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float64)
        done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)

        # now we are going to train the critics
        next_action = self.policy.get_target_action(state=next_state)

        target_value = self.critic.computeTargetValue(state=next_state, actions=next_action)

        expected_value = reward + (1 - done) * self.__gamma * tf.cast(target_value, dtype=tf.float64)

        critic_loss = self.critic.train(state=state, actions=actions, target=expected_value)

        # train the actor
        with tf.GradientTape() as tape:
            # critic the action take by the actor at the state s
            new_actions = self.policy.evaluate_state(state=state)
            new_actions = tf.convert_to_tensor(new_actions, dtype=tf.float64)
            policy_loss = self.critic.computeValue(state=state, actions=new_actions)
            policy_loss = -tf.math.reduce_mean(policy_loss)
        grads = tape.gradient(policy_loss, self.policy.policyNet.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.policyNet.trainable_variables))

        self.policy_losses.append(policy_loss)

        self.critic_losses.append(critic_loss)

        self.critic.softCopy()

        self.policy.softCopy()

    def save(self):
        model_filename = "save/Policy_Model" + self.game_name + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename = "save/Policy_Target_Model" + self.game_name + datetime.now().strftime("%Y%m%d-%H")
        self.policy.policyNet.save_weights(model_filename)
        self.policy.policyTargetNet.save_weights(target_model_filename)
        model_filename = "save/Critics_Model" + self.game_name + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename = "save/Critics_Target_Model" + self.game_name + datetime.now().strftime("%Y%m%d-%H")
        self.critic.valueNet.save(model_filename)
        self.critic.targetValueNet.save(target_model_filename)

    def computeAndWriteLoss(self, episode):
        policy_loss_mean = np.mean(self.policy_losses)
        critic_loss_mean = np.mean(self.critic_losses)
        tf.summary.scalar('Policy loss', policy_loss_mean, step=episode)
        tf.summary.scalar('Critics loss', critic_loss_mean, step=episode)
        print("Episode {} policy loss : {} critic loss : {}".format(episode, policy_loss_mean, critic_loss_mean))
        self.critic_losses.clear()
        self.policy_losses.clear()


def play():
    env_game = 'BipedalWalker-v2'
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + env_game + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    env = gym.make(env_game)
    scores = []
    agent = DDPGAgent(env=env, game_name=env_game)
    batch_size = 100
    training_delay = 100
    step = 0
    e = 0
    done = False
    s = env.reset()
    episode_iterations = 0
    ep_score = 0
    while step < 1e6:
        if done:
            with train_summary_writer.as_default():
                agent.computeAndWriteLoss(episode=e)
                tf.summary.scalar('episode_score', ep_score, step=e)
                scores.append(ep_score)
            agent.save()
            s = env.reset()
            e += 1
            episode_iterations = 0
            print("episode score {}".format(ep_score))
            print("Step {}".format(step))
            ep_score = 0
        # make the chosen action
        s = s.reshape([1, env.observation_space.shape[0]])
        observation = tf.convert_to_tensor(s, dtype=tf.float64)
        if training_delay < step:
            a = agent.get_action(state=observation, step=episode_iterations)
        else:
            a = env.action_space.sample()
        next_state, r, done, _ = env.step(a)
        agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)
        agent.updateNetworks(batch_size=batch_size, iterations=episode_iterations)
        ep_score += r
        s = next_state
        episode_iterations += 1
        step += 1
    return agent


def play2():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    env = gym.make('RoboschoolInvertedPendulum-v1')
    scores = []
    agent = DDPGAgent(env=env)
    batch_size = 100
    training_delay = 100
    e = 0
    episode_iterations = 0
    ep_score = 0
    for step in 100000:
        s = env.reset()
        for it in range(1000):
            s = s.reshape([1, env.observation_space.shape[0]])
            observation = tf.convert_to_tensor(s, dtype=tf.float64)
            if training_delay < step:
                a = agent.get_action(state=observation, step=episode_iterations)
            else:
                a = env.action_space.sample()
            next_state, r, done, _ = env.step(a)
            agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)
            agent.updateNetworks(batch_size=batch_size, iterations=episode_iterations)
            ep_score += r
            s = next_state
            if done:
                break
        with train_summary_writer.as_default():
            agent.computeAndWriteLoss(episode=e)
            tf.summary.scalar('episode_score', ep_score, step=e)
            scores.append(ep_score)
        agent.save()
        print("episode score {}".format(ep_score))
        print("Step {}".format(step))
        ep_score = 0
    return agent



if __name__ == '__main__':
    model = play()
    env = NormalizedActions(gym.make('BipedalWalker-v2'))
    env.reset()
    for e in range(1000):
        # reset the enviroment
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, env.observation_space.shape[0]])
            a = model.take_action(state=s)
            s, r, done, _ = env.step(a)
            env.render()