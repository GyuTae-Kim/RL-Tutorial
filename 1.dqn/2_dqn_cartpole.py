import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import gym


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
            tape.watch(target)
            Q = self.Q(tf.expand_dims(s, axis=0))
            loss = (Q[0][a] - target) ** 2
        grads = tape.gradient(loss, vars)
        self.opt.apply_gradients(zip(grads, vars))
        
        return loss
    

def train():
    env = gym.make('CartPole-v0')
    agent = Agent(env.action_space.n)
    episode = 2000
    r_list = []
    l_list = []
    
    for e in range(episode):
        state = env.reset()
        done = False
        epsilon = 1. / ((e // 10) + 1)
        step = 0
        R = 0
        L = 0.

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
            loss = agent.update(state, action, reward, new_state, done)
            L += loss
            state = new_state
        # -- end while -- #
        r_list.append(R)
        l_list.append(L / step)
        print(f'Episode {e}  steps: {step}  reward: {R}  loss: {L}')
    # -- end for -- #
    plt.plot(r_list)
    plt.xlabel('Episode')
    plt.ylabel('reward sum')
    plt.show()
    
    plt.plot(l_list)
    plt.xlabel('Episode')
    plt.ylabel('loss mean')
    plt.show()


if __name__ == '__main__':
    train()

