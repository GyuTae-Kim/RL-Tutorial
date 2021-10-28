import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import gym
import random
from collections import deque


MEMORY_SIZE = 20000


class Agent(object):
    
    def __init__(self, output_size):
        self.output_size = output_size
        
        self.Q = tf.keras.Sequential([
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(output_size, 'linear')
        ])
        self.opt = tf.keras.optimizers.Adam(1e-3)
        self.discount = .9

    def get_Q(self, state):
        return self.Q(tf.expand_dims(state, axis=0)).numpy()[0]
    
    def update(self, s, a, r, new_s, done, discount=.9):
        if done:
            target = tf.constant(r, dtype=tf.float32)
        else:
            target = tf.constant(r + discount * np.max(self.Q(tf.expand_dims(new_s, axis=0))), dtype=tf.float32)
        vars = self.Q.trainable_variables
        with tf.GradientTape() as tape:
            Q = self.Q(tf.expand_dims(s, axis=0))
            loss = (Q[0][a] - target) ** 2
        grads = tape.gradient(loss, vars)
        self.opt.apply_gradients(zip(grads, vars))
        
        return loss


def replay_train(agent, minibatch):
    loss = []
    for s, a, r, new_s, done in minibatch:
        loss.append(agent.update(s, a, r, new_s, done))
    
    return np.mean(loss)


def train():
    env = gym.make('CartPole-v1')
    agent = Agent(env.action_space.n)
    episode = 2000
    replay_buffer = deque()
    r_list = []
    l_list = []
    
    for e in range(episode):
        state = env.reset()
        done = False
        epsilon = 1. / ((e // 10) + 1)
        step = 0
        R = 0

        while not done:
            # get action
            if np.random.rand(1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(agent.get_Q(state))
            # move
            new_state, reward, done, _ = env.step(action)
            step += 1
            R += reward
            # update q-function
            replay_buffer.append((state, action, reward, new_state, done))
            if len(replay_buffer) > MEMORY_SIZE:
                replay_buffer.popleft()
            state = new_state
        # -- end while -- #
        print(f'Episode {e}  steps: {step}  reward: {R}')
        r_list.append(R)
        if e % 10 == 1:
            loss = 0.
            for _ in range(50):
                minibatch = random.sample(replay_buffer, 10)
                loss += replay_train(agent, minibatch)
            loss /= 50.
            print(f'Replay train loss: {loss}')
            l_list.append(loss)
        
    # -- end for -- #
    plt.plot(r_list)
    plt.xlabel('Episode')
    plt.ylabel('reward sum')
    plt.show()
    
    plt.plot(l_list)
    plt.xticks(np.arange(1, episode, 10))
    plt.xlabel('Episode')
    plt.ylabel('loss mean')
    plt.show()


if __name__ == '__main__':
    train()

