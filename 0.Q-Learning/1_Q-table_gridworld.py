from copy import deepcopy
import numpy as np


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
        self.Q = np.zeros([16, 4], dtype=np.float32) # (w * h), a
        self.define_dict = {
            'plain': 0,
            'heaven': 1,
            'hell': 2,
            'agent': 3
        }
        self.discount = .9

    def get_Q(self, state):
        return self.Q[state]
    
    def update(self, s, a, r, new_s, done, discount=.9):
        if done:
            self.Q[s, a] = r
        else:
            self.Q[s, a] = r + discount * np.max(self.Q[new_s, :])
    

def train():
    env = Environment()
    agent = Agent()
    episode = 200
    success = 0
    
    for e in range(episode):
        print(f'Episode {e + 1} start')
        state = np.argmax(env.reset())
        done = False
        epsilon = 1. / ((e // 10) + 1)

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
            new_state = np.argmax(new_state)
            # update q-table
            agent.update(state, action, reward, new_state, done)
            state = new_state
        # -- end while -- #
        if reward == 100.:
            print(f'Episode {e + 1} success, reward {reward}')
            success += 1
        else:
            print(f'Episode {e + 1} fail, reward {reward}')
    # -- end for -- #
    print('Success rate:', float(success / episode))
    print(agent.Q)


if __name__ == '__main__':
    train()
