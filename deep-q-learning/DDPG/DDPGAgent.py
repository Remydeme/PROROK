from DDPG.Networks.critics import Critics
from DDPG.Networks.policy import Policy
from DDPG.BufferReplay import ReplayBuffer

import tensorflow as tf
import numpy as np



class DDPGAgent():


    def __init__(self, env, gamma=0.99):
        self.env = env
        action_dim = env.action_space.shape[0]
        input_dim = env.observation_space.shape[0]
        self.policy = Policy(input_dim=input_dim, action_dim=action_dim, action_low=env.action_space.low, action_high=env.action_space.high)
        self.critic = Critics(env=env)
        self.replayBuffer = ReplayBuffer(obs_dim=input_dim, act_dim=action_dim, size=1000000)
        self.__gamma = gamma
        self.noise = OUNoise(env.action_space)



    def get_action(self, state, step):
        actions = self.policy.get_action(state=state)
        actions = self.noise.get_action(action=actions,t=step)
        return actions

    def take_action(self, state):
        actions = self.policy.get_action(state=state)
        return actions

    def updateNetworks(self, batch_size=32):
        state, next_state, actions, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

        # Keras want input of type tf.float64 by default in depend on the processor
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float64)
        done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)

        # train the actor
        with tf.GradientTape() as tape:
            # critic the action take by the actor at the state s
            new_actions = self.policy.evaluate_state(state=state)
            new_actions = tf.convert_to_tensor(new_actions, dtype=tf.float64)
            policy_loss = self.critic.computeValue(state=state, actions=new_actions)
            policy_loss = -tf.math.reduce_mean(policy_loss)
        print("policy loss : {}".format(policy_loss))
        grads = tape.gradient(policy_loss, self.policy.policyNet.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.policyNet.trainable_variables))

        # now we are going to train the critics
        next_action = self.policy.get_target_action(state=state)

        target_value = self.critic.computeTargetValue(state=next_state, actions=next_action)

        expected_value = reward  + self.__gamma * tf.cast(target_value, dtype=tf.float64)

        self.critic.train(state=state, actions=actions, target=expected_value)

        self.critic.softCopy()

        self.policy.softCopy()


    def updateCritics(self, batch_size=32):
        state, next_state, actions, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

        # Keras want input of type tf.float64 by default in depend on the processor
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float64)
        done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)

        # now we are going to train the critics

        next_action = self.policy.get_target_action(state=state)

        target_value = self.critic.computeTargetValue(state=next_state, actions=next_action)

        expected_value = reward + self.__gamma * tf.cast(target_value, dtype=tf.float64)

        self.critic.train(state=state, actions=actions, target=expected_value)

        self.critic.softCopy()

        self.policy.softCopy()


    def updatePolicy(self, batch_size=32):
        state, next_state, actions, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

        # Keras want input of type tf.float64 by default in depend on the processor
        state = tf.convert_to_tensor(state, dtype=tf.float64)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
        actions = tf.convert_to_tensor(actions, dtype=tf.float64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float64)
        done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)

        # train the actor
        with tf.GradientTape() as tape:
            # critic the action take by the actor at the state s
            new_actions = self.policy.evaluate_state(state=state)
            new_actions = tf.convert_to_tensor(new_actions, dtype=tf.float64)
            policy_loss = self.critic.computeValue(state=state, actions=new_actions)
            policy_loss = -tf.math.reduce_mean(policy_loss)
        print("policy loss : {}".format(policy_loss))
        grads = tape.gradient(policy_loss, self.policy.policyNet.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(grads, self.policy.policyNet.trainable_variables))

        self.policy.softCopy()




from DDPG.normalizedAction import NormalizedActions
import gym
from DDPG.UNoise import OUNoise
from datetime import  datetime

def play():
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    env = NormalizedActions(gym.make('MountainCarContinuous-v0'))
    scores = []
    agent = DDPGAgent(env=env)
    delay = 50
    batch_size = 32
    e = 0
    while e < 100:
        s = env.reset()
        ep_score = 0
        done = False
        start_step = 100
        episode_max_len = 1000
        agent.noise.reset()
        for step in range(500):
            # make the chosen action
            s = s.reshape([1, env.observation_space.shape[0]])
            observation = tf.convert_to_tensor(s, dtype=tf.float64)
            a = agent.get_action(state=observation, step=step)
            # print("actions : {}".format(a))
            next_state, r, done, _ = env.step(a)
            agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)
            if agent.replayBuffer.size >= batch_size:
                agent.updateNetworks(batch_size=batch_size)
            ep_score += r
            s = next_state
            if done == True:
                break
        print("Episode score {}".format(ep_score))
        scores.append(ep_score)

        if e % delay == 0:
            # compute the mean score on the 10 last episode
            print("***------------Episode  {}  Score  {}-------------*****".format(e, np.mean(scores[-delay:])))
            with train_summary_writer.as_default():
                tf.summary.scalar('mean_score', np.mean(scores[-delay:]), step=e)
        # save the model every 100 episode
        # agent.save_model()
        if np.mean(scores[-delay:]) >= 90:
            break
        e += 1
    return agent




if __name__ == "__main__":
    model = play()
    env = NormalizedActions(gym.make('MountainCarContinuous-v0'))
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



