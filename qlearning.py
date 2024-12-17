### QLearning
import numpy as np
import pandas as pd
import pygame
from rl_game.racegame import RaceEnv, MAX_SPEED, MIN_SPEED, ACCELERATION, TURN_ACCELERATION,VIEW
from rl_game.helpers import get_discrete_state
import sys

### Script / Visualization Settings
LOAD_QTABLE = "load" in sys.argv
SAVE_QTABLE = "save" in sys.argv
START_SHOWING_FROM = 0 #400
SHOW_EVERY = 10

### Training settings
LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 2000#000 #10000
START_RANDOM_STARTING_FROM = 0#200#1500

### Exploration settings
epsilon = 0.5 # not a constant, qoing to be decayed
EPSILON_MIN = 0.001 #Noise injection: even when decay is over, leave some exploration to avoid being stuck in local minima
START_EPSILON_DECAYING = 0#1000
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = (epsilon-EPSILON_MIN)/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

### STATES & ACTIONS: create bins of continous states for the Q-table
# distances of 8 whiskers
distances_bins = [np.linspace(0, VIEW, 4)]*8
# for direciton and speed, needs to be designed such that each action ends up in a new bin
direction_bins = np.linspace(0, 360, int(360/TURN_ACCELERATION+1))
speed_bins = np.linspace(MIN_SPEED, MAX_SPEED, int((abs(MIN_SPEED)+MAX_SPEED)/ACCELERATION+1)) #ideally: 20
all_bins = distances_bins + [direction_bins] + [speed_bins]
# action space
# Game controls: 0: no action (e.g. keep going straight), 1: forward, 2: backward/brake, 3: left, 4: right
# For qlearning, we drop no action [0,0] to prevent the agent from getting stuck. has to either move or turn
actions = [(1,0), (-1,0), (0,1), (0,-1)]

logging_cols = ["Steps,", "Epsilon", "Cumulative Q", "Cumulative Reward", "Max Q", "Min Q", "P90 Q","Max R", "Min R","P90 R", "endX", "endY"]
### INITIALIZE Q-TABLE
if LOAD_QTABLE:
    print("LOADING Q-TABLE")
    q_table = np.load("results/q_table.npy")
    logging_arr = pd.read_feather("results/logging.feather").values
else:
    # Q-table: state space x action space
    table_size = [len(bin)-1 for bin in all_bins] + [len(actions)]
    q_table = np.random.uniform(low=-6.5, high=-5.5, size= table_size)
    # logging
    logging_arr = np.array(len(logging_cols))


# n cells: 
print(f"Q-table size: {q_table.shape}")
num_values = np.prod(q_table.shape)
print(f"Total number of values in Q-table: {num_values}")


### QLEARNING LOOP

# initialize environment
environment = RaceEnv()
environment.init_render()
render = False
random_start = False # Only start at random position after n episodes

# Q-learning loop
for episode in range(EPISODES):
    if (episode % SHOW_EVERY == 0):#only plot every n episodes after initial episodes are over
        print(f"Episode: {episode}, epsilon: {round(epsilon,5)}")
        if episode >= START_SHOWING_FROM:
            render = True
    random_start = episode >= START_RANDOM_STARTING_FROM
    done = False
    checkpoint_reached = False
    steps = 0
    Q = []
    R = []
    # get initial state
    discrete_state = get_discrete_state(environment.reset(random_start = random_start), all_bins)

    while not done:
        # Get action: exploit or explore
        if np.random.random() > epsilon:
            # Exploit: Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Explore: Get random action
            action = np.random.randint(0, len(actions))
        # perform
        new_state, reward, done, checkpoint_reached = environment.step(actions[action])
        print("Checkpoint reached!") if checkpoint_reached else None
        new_discrete_state = get_discrete_state(new_state, all_bins)
        #print(f"New state: {new_state}")
        #print(f"New state discrete: {new_discrete_state}")
        # render current state
        if render:
            environment.render()

        ### UPDATE Q TABLE
        # check if episode is over or checkpoint reached. If so, set Q directly to reward
        if done or checkpoint_reached:
            new_q = reward
        else:
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
    episode_log = [steps, epsilon, np.sum(Q), np.sum(R), np.max(Q), np.min(Q),np.quantile(Q, 0.9), np.max(R), np.min(R),np.quantile(R, 0.9), environment.car.rect.center[0], environment.car.rect.center[1]]
    np.concatenate([logging_arr, episode_log])

    # every 100 episodes, save q_table + results
    if episode % 100 == 0 and SAVE_QTABLE:
        np.save("results/q_table.npy", q_table)
        pd.DataFrame(logging_arr, columns=logging_cols).to_feather("results/logging.feather")

if SAVE_QTABLE:
    # save final Q-table and logging
    np.save("results/q_table.npy", q_table)
    pd.DataFrame(logging_arr, columns=logging_cols).to_feather("results/logging.feather")

pygame.quit()
