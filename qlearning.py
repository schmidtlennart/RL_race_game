### QLearning
import numpy as np
import random
from env import RaceEnv

MAX_SPEED = 8
WINDOW_SIZE = 1020, 770

def create_bins(min_value, max_value, n_bins):
    bins = []
    bin_size = (max_value - min_value + 1) / n_bins
    current_value = min_value
    for i in range(n_bins):
        next_value = min(current_value + bin_size - 1, max_value)
        bins.append((int(current_value), int(next_value)))
        current_value += bin_size
    return bins

# get min and max values for observations
direction_bins = create_bins(0, 359, 4)
speed_bins = create_bins(-MAX_SPEED, MAX_SPEED, 4)
position_x_bins = create_bins(0, WINDOW_SIZE[0], 10)
position_y_bins = create_bins(0, WINDOW_SIZE[1], 10)

# action space
actions = [0,1, 2, 3, 4]  # 0: no action (e.g. keep going straight), 1: forward, 2: backward, 3: left, 4: right


obs_space = 20
action_space = 4
q_table = np.zeros((obs_space, obs_space, action_space))




environment = RaceEnv()
environment.init_render()
run = True

while run:
    # set game speed to 30 fps
    environment.clock.tick(30)
    # ─── CONTROLS ───────────────────────────────────────────────────────────────────
    # get pressed keys, generate action
    action = environment.pressed_to_action()
    # calculate one step
    new_state, reward, done = environment.step(action)
    # render current state
    environment.render()
pygame.quit()


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