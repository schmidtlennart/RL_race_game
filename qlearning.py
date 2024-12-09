### QLearning
import numpy as np
import pandas as pd
import pygame
from rl_game.racegame import RaceEnv, MAX_SPEED, VIEW
from rl_game.helpers import get_discrete_state
import sys

LEARNING_RATE = 0.2
DISCOUNT = 0.95
EPISODES = 10000 #10000
START_SHOWING_FROM = 1000 #1000
SHOW_EVERY = 50
LOAD_QTABLE = sys.argv[1] == "load"

# Exploration settings
epsilon = 0.005#0.1  # not a constant, qoing to be decayed
EPSILON_MIN = 0.001 #Noise injection: even when decay is over, leave some exploration if stuck in local minima
START_EPSILON_DECAYING = 200
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = (epsilon-EPSILON_MIN)/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

### STATES & ACTIONS: create bins of continous states for the Q-table
# distances of 8 whiskers
distances_bins = [np.linspace(0, VIEW, 4)]*8 #ideally : 30
direction_bins = np.linspace(0, 359, 36)
speed_bins = np.linspace(-MAX_SPEED, MAX_SPEED, 6) #ideally: 20
all_bins = distances_bins + [direction_bins] + [speed_bins]
# action space
actions = [(0,0),(1,0), (-1,0), (0,1), (0,-1)]  # 0: no action (e.g. keep going straight), 1: forward, 2: backward/brake, 3: left, 4: right

# initialize Q-table
if LOAD_QTABLE:
    print("LOADING Q-TABLE")
    q_table = np.load("results/q_table.npy")
else:
    table_size = [len(bin)-1 for bin in all_bins] + [len(actions)]
    q_table = np.random.uniform(low=-6.5, high=-5.5, size= table_size)


### QLEARNING LOOP

# initialize environment
environment = RaceEnv()
environment.init_render()
render = False
logging_cols = ["Episode", "Epsilon", "Steps,", "Cumulative Q", "Cumulative Reward", "Max Q", "Min Q", "P90 Q","Max R", "Min R","P90 R", "endX", "endY"]
logging_arr = np.full((EPISODES, len(logging_cols)), np.nan)

for episode in range(EPISODES):
    if (episode % SHOW_EVERY == 0):#only plot every n episodes after initial episodes are over
        print(f"Episode: {episode}, epsilon: {round(epsilon,5)}")
        if episode >= START_SHOWING_FROM:
            render = True
    done = False
    steps = 0
    Q = []
    R = []
    # get initial state
    discrete_state = get_discrete_state(environment.reset(random_start = True), all_bins)

    while not done:
        # Get action: exploit or explore
        if np.random.random() > epsilon:
            # Exploit: Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Explore: Get random action
            action = np.random.randint(0, len(actions))
        # perform
        new_state, reward, done = environment.step(actions[action])
        new_discrete_state = get_discrete_state(new_state, all_bins)
        #print(f"New state: {new_state}")
        #print(f"New state discrete: {new_discrete_state}")
        # render current state
        if render:
            environment.render()

        ### UPDATE Q TABLE
        # Intuition: Update current Q value with the maximum Q value that could be reached after the action
        # Maximum possible Q value in next step (for new state)
        max_future_q = np.max(q_table[new_discrete_state])

        # Current Q value (for current state and performed action)
        current_q = q_table[discrete_state + (action,)]

        # And here's our equation for a new Q value for current state and action
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # Update Q table with new Q value
        q_table[discrete_state + (action,)] = new_q
        
        # Update current state for new loop
        discrete_state = new_discrete_state

        # log
        Q.append(new_q)
        R.append(reward)
        steps += 1

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    render = False
    # log
    logging_arr[episode, :] = [episode, steps, epsilon, np.sum(Q), np.sum(R), np.max(Q), np.min(Q),np.quantile(Q, 0.9), np.max(R), np.min(R),np.quantile(R, 0.9), environment.car.rect.center[0], environment.car.rect.center[1]]


# save Q-table
np.save("results/q_table.npy", q_table)
pygame.quit()

# save log as pd df
pd.DataFrame(logging_arr, columns=logging_cols).to_feather("results/logging.feather", index=False)
