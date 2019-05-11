from D3PG.Networks.Critics import Critics
from D3PG.Networks.Policy import Policy
from D3PG.replayBuffer import ReplayBuffer
import roboschool
import tensorflow as tf
import numpy as np


from datetime import datetime

from D3PG.normalizedAction import NormalizedActions

import gym


class D3PGAgent():

    def __init__(self, env, gamma=0.995, target_noise=0.2, actor_noise=0.1):

        self.env = env
        self.__target_noise = target_noise
        self.__actor_noise = actor_noise
        action_dim = env.action_space.shape[0]
        input_dim = env.observation_space.shape[0]
        self.policy = Policy(input_dim=input_dim, action_dim=action_dim, action_low=env.action_space.low,
                             action_high=env.action_space.high)
        self.critic = Critics(env=env)
        self.criticBis = Critics(env=env)
        self.replayBuffer = ReplayBuffer(obs_dim=input_dim, act_dim=action_dim, size=1000000)
        self.__gamma = gamma
        self.policy_losses = []
        self.critic_losses = []

    def get_action(self, state):

        actions = self.policy.get_action(state=state) + np.random.normal(0, self.__actor_noise, size=self.env.action_space.shape[0])
        actions = tf.clip_by_value(actions, self.env.action_space.low, self.env.action_space.high)
        return actions

    def take_action(self, state):

        actions = self.policy.get_action(state=state)
        return actions



    def update(self, batch_size=100, iterations=100):

        for step in range(iterations):
            self.updateCritics(batch_size=batch_size)
            if step % 2:
                self.updatePolicy(batch_size=batch_size)
                self.copyNetworksToTarget()

    def updatePolicy(self, batch_size=100):

        # take a random batch of elements
        state, _, _, _, _ = self.replayBuffer.sample_batch(batch_size=batch_size)

        # convert our numpy array into tensor of float64
        state = tf.convert_to_tensor(state, dtype=tf.float64)

        # train the actor
        with tf.GradientTape() as tape:
            # critic the action take by the actor at the state s
            new_actions = self.policy.evaluate_state(state=state)
            new_actions = tf.convert_to_tensor(new_actions, dtype=tf.float64)
            qValue = self.critic.computeValue(state=state, actions=new_actions)
            policy_loss = -tf.math.reduce_mean(qValue)
            self.policy_losses.append(policy_loss)

        # apply the computed gradient
        grads = tape.gradient(policy_loss, self.policy.policyNet.trainable_variables)

        self.policy.optimizer.apply_gradients(zip(grads, self.policy.policyNet.trainable_variables))

    def updateCritics(self, batch_size=100):

        state, next_state, actions, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

        # Keras want input of type tf.float64 by default in depend on the processor
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float64)
        done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)

        with tf.GradientTape(persistent=True) as tape:

            q_value = self.critic.computeValue(state=state, actions=actions)

            q_value_bis = self.criticBis.computeValue(state=state, actions=actions)
            # now we are going to train the critics

            next_actions = self.policy.get_target_action(state=state)

            next_actions = next_actions + np.clip(np.random.normal(0, self.__target_noise, size=self.env.action_space.shape[0]), -0.5, 0.5)

            next_actions = np.clip(next_actions, self.env.action_space.low, self.env.action_space.high)

            target_value = self.critic.computeTargetValue(state=next_state, actions=next_actions)

            target_value_bis = self.criticBis.computeTargetValue(state=next_state, actions=next_actions)

            target_value = tf.math.minimum(target_value, target_value_bis)

            expected_value = reward + (1 - done) * self.__gamma * tf.cast(target_value, dtype=tf.float64)

            critic_loss = self.critic.computeLosses(value=q_value, target=expected_value)


            critic_loss_bis = self.criticBis.computeLosses(value=q_value_bis, target=expected_value)


            loss = critic_loss + critic_loss_bis

            self.critic_losses.append(loss)

        grad = tape.gradient(loss, self.critic.valueNet.trainable_variables)

        grad_bis = tape.gradient(loss, self.criticBis.valueNet.trainable_variables)

        self.critic.optimizer.apply_gradients(zip(grad, self.critic.valueNet.trainable_variables))

        self.criticBis.optimizer.apply_gradients(zip(grad_bis, self.criticBis.valueNet.trainable_variables))

    def copyNetworksToTarget(self):

        self.policy.softCopy()

        self.critic.softCopy()

        self.criticBis.softCopy()

    def save(self):

        model_filename = "save/Policy_Model" + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename = "save/Policy_Target_Model" + datetime.now().strftime("%Y%m%d-%H")
        self.policy.policyNet.save_weights(model_filename)
        self.policy.policyTargetNet.save_weights(target_model_filename)
        model_filename = "save/Critics_Model" + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename = "save/Critics_Target_Model" + datetime.now().strftime("%Y%m%d-%H")
        self.critic.valueNet.save(model_filename)
        self.critic.targetValueNet.save(target_model_filename)
        model_filename_bis = "save/Critics_Model_Bis" + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename_bis = "save/Critics_Target_Model_Bis" + datetime.now().strftime("%Y%m%d-%H")
        self.criticBis.valueNet.save(model_filename_bis)
        self.criticBis.targetValueNet.save(target_model_filename_bis)

    def computeAndWritePolicyLoss(self, episode):

        policy_loss_mean = np.mean(self.policy_losses)
        critic_loss_mean = np.mean(self.critic_losses)
        tf.summary.scalar('Policy loss', policy_loss_mean, step=episode)
        tf.summary.scalar('Critics loss', critic_loss_mean, step=episode)
        print("Episode {} policy loss : {} critic loss : {}".format(episode, policy_loss_mean, critic_loss_mean))
        self.critic_losses.clear()
        self.policy_losses.clear()



def play():
    current_time = datetime.now().strftime("%Y%m%d-%H")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    env = NormalizedActions(gym.make('RoboschoolAnt-v1'))
    scores = []
    agent = D3PGAgent(env=env)
    batch_size = 100
    training_delay = 10000
    step = 0
    e = 0
    done = False
    s = env.reset()
    episode_iterations = 0
    ep_score = 0
    ep_len_max = 1000
    while step < 1e6:
        if done:
            agent.update(batch_size=batch_size, iterations=episode_iterations)
            with train_summary_writer.as_default():
                agent.computeAndWritePolicyLoss(episode=e)
                tf.summary.scalar('episode_score', ep_score, step=e)
                scores.append(ep_score)
            print("Step {}".format(step))
            print("episode score {}".format(ep_score))
            ep_score = 0
            agent.save()
            s = env.reset()
            e += 1
            episode_iterations = 0
        # make the chosen action
        s = s.reshape([1, env.observation_space.shape[0]])
        observation = tf.convert_to_tensor(s, dtype=tf.float64)
        if training_delay < step:
            a = agent.get_action(state=observation)
        else:
            a = env.action_space.sample()
        next_state, r, done, _ = env.step(a)
        agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)
        ep_score += r
        s = next_state
        episode_iterations += 1
        step += 1
        if episode_iterations >= ep_len_max:
            done = True

    return agent


if __name__ == "__main__":
    """

    RoboschoolHalfCheetah-v1

    """
    model = play()
    env = NormalizedActions(gym.make('RoboschoolAnt-v1'))
    env.reset()
    for e in range(100):
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