### QLearning


import numpy as np
import random


obs_space = 20
action_space = 4
q_table = np.zeros((obs_space, obs_space, action_space))


# class QLearning:
#     def __init__(self, actions, epsilon, alpha, gamma):
#         self.q = {}
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.gamma = gamma
#         self.actions = actions

#     def getQ(self, state, action):
#         return self.q.get((state, action), 0.0)

#     def chooseAction(self, state):
#         if random.random() < self.epsilon:
#             action = random.choice(self.actions)
#         else:
#             q = [self.getQ(state, a) for a in self.actions]
#             maxQ = max(q)
#             count = q.count(maxQ)
#             if count > 1:
#                 best = [i for i in range(len(self.actions)) if q[i] == maxQ]
#                 i = random.choice(best)
#             else:
#                 i = q.index(maxQ)
#             action = self.actions[i]
#         return action

#     def learn(self, state, action, reward, value):
#         oldv = self.q.get((state, action), None)
#         if oldv is None:
#             self.q[(state, action)] = reward
#         else:
#             self.q[(state, action)] = oldv + self.alpha * (value - oldv)

#     def update(self, state, action, reward, new_state):
#         q = [self.getQ(new_state, a) for a in self.actions]
#         futureReward = max(q)
#         self.learn(state, action, reward, reward + self.gamma * futureReward)

#     def save(self, filename):
#         with open(filename, 'w') as f:
#             for k, v in self.q.items():
#                 f.write(str(k[0][0]) + ',' + str(k[0][1]) + ',' + str(k[1]) + ',' + str(v) + '\n')

#     def load(self, filename):
#         with open(filename, 'r') as f:
#             for line in f:
#                 parts = line.split(',')
#                 self.q[(int(parts[0]), int(parts[1]), int(parts[2]))] = float(parts[3])