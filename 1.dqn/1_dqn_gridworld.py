from copy import deepcopy
import numpy as np
import tensorflow as tf


IMPOS = -np.Inf

class Environment(object):
    
    def __init__(self):
        self.define_dict = {
            'plain': 0,
            'heaven': 1,
            'hell': 2,
            'agent': 3
        }
        self.init_map = [
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            2, 0, 0, 1
        ]
    
    def reset(self):
        self.end = False
        self.agent_idx = 0
        self.state = np.array(deepcopy(self.init_map))
        self.state[self.agent_idx] = self.define_dict['agent']
        return deepcopy(self.state)

    def step(self, action):
        if self.end:
            raise NotImplementedError('Environment must be reset.')
        
        if action == 0:     # 동
            next_idx = self.agent_idx + 1
        elif action == 1:   # 서
            next_idx = self.agent_idx - 1
        elif action == 2:   # 남
            next_idx = self.agent_idx + 4
        else:               # 북
            next_idx = self.agent_idx - 4
        self.state[self.agent_idx] = self.define_dict['plain']
        
        if self.state[next_idx] == self.define_dict['plain']:
            self.state[next_idx] = self.define_dict['agent']
            self.agent_idx = next_idx
            r = 0.
        elif self.state[next_idx] == self.define_dict['heaven']:
            self.agent_idx = next_idx
            self.end = True
            r = 100.
        else:   # hell
            self.agent_idx = next_idx
            self.end = True
            r = -100.
            
        return deepcopy(self.state), r, self.end
    
    def available_action(self):
        actions = []
        if self.agent_idx % 4 != 3:     # 동
            actions += [0]
        if self.agent_idx % 4 != 0:     # 서
            actions += [1]
        if self.agent_idx // 4 != 3:    # 남
            actions += [2]
        if self.agent_idx // 4 != 0:    # 북
            actions += [3]
        
        return actions


class Agent(object):
    
    def __init__(self):
        self.Q = tf.keras.Sequential([
            tf.keras.layers.Dense(16),
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(4, 'linear')
        ])
        self.opt = tf.keras.optimizers.Adam(1e-3)
        self.define_dict = {
            'plain': 0,
            'heaven': 1,
            'hell': 2,
            'agent': 3
        }
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
    env = Environment()
    agent = Agent()
    episode = 200
    success = 0
    
    for e in range(episode):
        print(f'Episode {e + 1} start')
        state = env.reset()
        done = False
        epsilon = 1. / ((e // 10) + 1)
        step = 0

        while not done:
            # get action
            aa = env.available_action()
            if np.random.rand(1) < epsilon:
                action = np.random.choice(aa)
            else:
                Q = agent.get_Q(state)
                action = np.argmax([q if i in aa else IMPOS for i, q in enumerate(Q)])
            # move
            new_state, reward, done = env.step(action)
            step += 1
            # update q-function
            loss = agent.update(state, action, reward, new_state, done)
            state = new_state
            # print(f'step: {step} loss: {loss}')
        # -- end while -- #
        if reward == 100.:
            print(f'Episode {e + 1} success')
            success += 1
        else:
            print(f'Episode {e + 1} fail')
    # -- end for -- #
    print('Success rate:', float(success / episode))
    s = np.array(deepcopy(env.init_map), dtype=np.int).reshape(1, -1)
    s = s * np.ones((s.shape[1], 1), dtype=np.int)
    diag = np.arange(0, s.shape[0], dtype=np.int)
    s[diag, diag] = 3
    print(agent.Q(tf.convert_to_tensor(s, dtype=tf.float32)).numpy())


if __name__ == '__main__':
    train()
