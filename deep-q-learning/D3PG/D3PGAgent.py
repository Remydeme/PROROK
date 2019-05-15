from D3PG.Networks.Critics import Critics
from D3PG.Networks.Policy import Policy
from D3PG.replayBuffer import ReplayBuffer
import roboschool
import tensorflow as tf
import numpy as np


from datetime import datetime


import gym


class D3PGAgent():

    def __init__(self, env, env_name, gamma=0.99, policy_noise=0.2, actor_noise=0.1):

        self.env = env
        self.env_name = env_name
        self.__policy_noise = policy_noise
        self.actor_noise = actor_noise
        self.action_max = self.env.action_space.high[0]
        action_dim = env.action_space.shape[0]
        input_dim = env.observation_space.shape[0]
        self.policy = Policy(action_dim=action_dim, action_high=env.action_space.high)
        self.critic_Q1 = Critics(env=env)
        self.critic_Q2 = Critics(env=env)
        self.replayBuffer = ReplayBuffer(obs_dim=input_dim, act_dim=action_dim, size=10000000)
        self.__gamma = gamma
        self.policy_losses = []
        self.critic_losses = []

    def get_action(self, state):

        actions = self.policy.get_action(state=state) + np.random.normal(0, self.actor_noise, size=self.env.action_space.shape[0])
        actions = tf.clip_by_value(actions, -self.action_max, self.action_max)
        return actions.numpy()

    def take_action(self, state):
        actions = self.policy.get_action(state=state)
        return actions.numpy()

    def update(self, batch_size=100, iterations=100, noise_clip=0.5):

        for step in range(iterations):

            state, next_state, actions, reward, done = self.replayBuffer.sample_batch(batch_size=batch_size)

            # Keras want input of type tf.float64 by default in depend on the processor
            state = tf.convert_to_tensor(state, dtype=tf.float64)
            next_state = tf.convert_to_tensor(next_state, dtype=tf.float64)
            actions = tf.convert_to_tensor(actions, dtype=tf.float64)
            reward = tf.convert_to_tensor(reward, dtype=tf.float64)
            done = tf.convert_to_tensor(np.float64(done), dtype=tf.float64)


            noise = np.clip(np.random.normal(0, self.__policy_noise), -noise_clip, noise_clip)

            next_actions = self.policy.get_target_action(state=(next_state + noise))

            next_actions = np.clip(next_actions, -self.action_max, self.action_max)


            # compute Q(A', S')
            target_Q1 = self.critic_Q1.computeTargetValue(state=next_state, actions=next_actions)

            target_Q2 = self.critic_Q2.computeTargetValue(state=next_state, actions=next_actions)

            target_value = tf.math.minimum(target_Q1, target_Q2)

            # compute the target
            expected_value = reward + (1 - done) * self.__gamma * tf.cast(target_value, dtype=tf.float64)


            # compute Q(S, A) and QBis(S, A)

            critic_loss_Q1 = self.critic_Q1.train(state=state, actions=actions, target=expected_value)

            critic_loss_Q2 = self.critic_Q2.train(state=state, actions=actions, target=expected_value)

            critics_loss = critic_loss_Q1 + critic_loss_Q2

            self.critic_losses.append(critics_loss)



            if iterations % 2 == 0:

                # train the actor
                with tf.GradientTape() as tape:
                    # critic the action take by the actor at the state s
                    new_actions = self.policy.evaluate_state(state=state)
                    qValue = self.critic_Q1.computeValue(state=state, actions=new_actions)
                    policy_loss = - tf.math.reduce_mean(qValue)
                    self.policy_losses.append(policy_loss)

                # apply the computed gradient
                policy_grads = tape.gradient(policy_loss, self.policy.policyNet.trainable_variables)

                self.policy.optimizer.apply_gradients(zip(policy_grads, self.policy.policyNet.trainable_variables))

                self.copyNetworksToTarget()

    def copyNetworksToTarget(self):

        self.policy.softCopy()

        self.critic_Q1.softCopy()

        self.critic_Q2.softCopy()

    def save(self):

        model_filename = "save/Policy_Model" + self.env_name + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename = "save/Policy_Target_Model" + self.env_name + datetime.now().strftime("%Y%m%d-%H")
        self.policy.policyNet.save_weights(model_filename)
        self.policy.policyTargetNet.save_weights(target_model_filename)
        model_filename = "save/Critics_Model" + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename = "save/Critics_Target_Model" + self.env_name + datetime.now().strftime("%Y%m%d-%H")
        self.critic_Q1.valueNet.save(model_filename)
        self.critic_Q1.targetValueNet.save(target_model_filename)
        model_filename_bis = "save/Critics_Model_Bis" + self.env_name + datetime.now().strftime("%Y%m%d-%H")
        target_model_filename_bis = "save/Critics_Target_Model_Bis" + self.env_name + datetime.now().strftime("%Y%m%d-%H")
        self.critic_Q2.valueNet.save(model_filename_bis)
        self.critic_Q2.targetValueNet.save(target_model_filename_bis)

    def computeAndWritePolicyLoss(self, episode):

        policy_loss_mean = np.sum(self.policy_losses)
        critic_loss_mean = np.sum(self.critic_losses)
        tf.summary.scalar('Policy loss', policy_loss_mean, step=episode)
        tf.summary.scalar('Critics loss', critic_loss_mean, step=episode)
        print("Episode {} policy loss : {} critic loss : {}".format(episode, policy_loss_mean, critic_loss_mean))
        self.critic_losses.clear()
        self.policy_losses.clear()

    def load_model(self, actor_name):
        self.policy.policyNet.load_weights(actor_name)


def evaluateModel(env, agent, episode):
    for e in range(10):
        s = env.reset()
        ep_memory = []
        ep_score = 0
        done = False
        while not done:
            s = s.reshape([1, env.observation_space.shape[0]])
            a = agent.take_action(state=s)
            s, r, done, _ = env.step(a)
            ep_score += r
        ep_memory.append(ep_score)
    evaluation_score = np.mean(ep_memory)
    tf.summary.scalar('evaluation score', evaluation_score, step=episode)




def play(env, agent):
    current_time = datetime.now().strftime("%Y%m%d-%H")
    train_log_dir = 'logs/gradient_tape/' + "Agent-mean-score-with-baseline" + current_time
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    scores = []
    batch_size = 100
    training_delay = 1000
    step = 0
    e = 0
    done = False
    s = env.reset()
    episode_iterations = 0
    ep_score = 0
    evaluation_delay = 100
    ep_max_len = 1000
    while step < 1e6:
        if done:
            agent.update(batch_size=batch_size, iterations=episode_iterations)
            with train_summary_writer.as_default():
                agent.computeAndWritePolicyLoss(episode=e)
                tf.summary.scalar('episode score', ep_score, step=e)
                scores.append(ep_score)
                if e % evaluation_delay == 0 and e != 0:
                    evaluateModel(env=env, agent=agent, episode=e)
            print("episode {} | step {}".format(e, step))
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
        ep_score += r
        agent.replayBuffer.store(obs=s, act=a, rew=r, next_obs=next_state, done=done)
        s = next_state
        if episode_iterations >= ep_max_len:
            done = True
        episode_iterations += 1
        step += 1
    return agent


def testModel(env):
    model = D3PGAgent(env=env)
    model.load_model(actor_name='save/Policy_Model20190513-11')
    env.reset()
    for e in range(1000):
        # reset the enviroment
        s = env.reset()
        done = False
        while not done:
            s = s.reshape([1, env.observation_space.shape[0]])
            s = tf.cast(s, dtype=tf.float64)
            a = model.take_action(state=s)
            s, r, done, _ = env.step(a.numpy())
            env.render()

if __name__ == "__main__":
    env_game = 'RoboschoolHalfCheetah-v1'
    env = gym.make(env_game)
    agent = D3PGAgent(env=env)
    model = play(env=env, agent=agent)
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
            s, r, done, _ = env.step(a.numpy())
            env.render()