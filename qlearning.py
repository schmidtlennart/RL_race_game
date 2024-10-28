### QLearning
import numpy as np
import pygame
from rl_game.racegame import RaceEnv, MAX_SPEED, WINDOW_WIDTH, WINDOW_HEIGHT
from rl_game.helpers import get_discrete_state

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 50

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# create bins for the Q-table
# position_x_bins = np.linspace(-20, WINDOW_WIDTH+20, 41) #needs a bit extra space, position can be negative/slightly outside of window
# position_y_bins = np.linspace(-20, WINDOW_HEIGHT+20, 41)

# position is now distances in 4 dimensions
distance_l = np.linspace(0, 300, 20)
distance_r = np.linspace(0, 300, 20)
distance_t = np.linspace(0, 300, 20)
distance_b = np.linspace(0, 300, 20)

direction_bins = np.linspace(0, 359, 36)
speed_bins = np.linspace(-MAX_SPEED-0.2, MAX_SPEED+0.2, 9) # extra to catch boundaries
all_bins = [distance_l,distance_r, distance_t, distance_b, direction_bins, speed_bins]
# action space
actions = [(0,0),(1,0), (-1,0), (0,1), (0,-1)]  # 0: no action (e.g. keep going straight), 1: forward, 2: backward, 3: left, 4: right
# initialize Q-table
q_table = np.random.uniform(low=-0.5, high=0, size= [len(distance_l)-1, len(distance_r)-1, len(distance_t)-1, len(distance_b)-1, len(direction_bins)-1, len(speed_bins)-1, len(actions)])

q_table.size

### QLEARNING LOOP

# initialize environment
environment = RaceEnv()
environment.init_render()
render = False

for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}")
        render = True
    done = False
    # get initial state
    discrete_state = get_discrete_state(environment.reset(), all_bins)

    while not done:
        # set game speed to 30 fps
        #environment.clock.tick(30)

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
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        # Inuition: Update current Q value with the maximum Q value that could be reached after the action
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

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    render = False
pygame.quit()
