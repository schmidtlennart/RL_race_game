### QLearning
import numpy as np
import pandas as pd
import pygame
from rl_game.racegame import RaceEnv
from rl_game.helpers import get_discrete_state, calc_bins
from rl_game.deepq_learners import DeepQLearner
import torch
import sys

### Script / Visualization Settings
LOAD = "load" in sys.argv
SAVE = "save" in sys.argv
FOLDER_NAME = "deepq_nn"

START_SHOWING_FROM = 50 #400
SHOW_EVERY = 1

### Training settings
LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 2000#000 #10000
START_RANDOM_STARTING_FROM = 0#200#1500

### Exploration settings
epsilon = 0.1 # not a constant, qoing to be decayed
EPSILON_MIN = 0 #Noise injection: even when decay is over, leave some exploration to avoid being stuck in local minima
START_EPSILON_DECAYING = 0#1000
END_EPSILON_DECAYING = 500
epsilon_decay_value = (epsilon-EPSILON_MIN)/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
# Increase epsilon by delta after n episodes
EPSILON_PULSE = 0.02
EPSILON_PULSE_AT = 700


### STATES & ACTIONS: create bins of continous states for the Q-table
all_bins = calc_bins()

# action space
# Game controls: 0: no action (e.g. keep going straight), 1: forward, 2: backward/brake, 3: left, 4: right
# For qlearning, we drop no action [0,0] to prevent the agent from getting stuck. has to either move or turn
actions = [(1,0), (-1,0), (0,1), (0,-1)]

logging_cols = ["Steps", "Epsilon", "Cumulative Q", "Cumulative Reward", "Cumulative TD-Error", 
                "Max Q", "Min Q", "P90 Q","Max R", "Min R","P90 R", "endX", "endY"]
#logging_cols = ["Steps", "Epsilon", "Cumulative Q", "Cumulative Reward", "Cumulative TD-Error", "P10 TDE","P90 TDE","Max Q", "Min Q", "P90 Q","Max R", "Min R","P90 R", "endX", "endY"]

    

### INITIALIZE AGENT

if LOAD:
    print("LOADING Q-MODELS")
    q_agent = DeepQLearner(state_size=10, action_size=len(actions), seed=36, 
                           buffer_size=10000, batch_size=64, discount=DISCOUNT, lr=0.001, tau=0.001, update_every=4)
    q_agent.qnetwork_local.load_state_dict(torch.load(f"results/{FOLDER_NAME}/qnetwork_local.pth"))
    q_agent.qnetwork_local.eval()
    q_agent.qnetwork_target.load_state_dict(torch.load(f"results/{FOLDER_NAME}/qnetwork_target.pth"))
    q_agent.qnetwork_target.eval()
    logging_list = pd.read_feather(f"results/{FOLDER_NAME}/logging.feather").values.tolist()
else:
    q_agent = DeepQLearner(state_size=10, action_size=len(actions), seed=36, 
                           buffer_size=10000, batch_size=64, discount=DISCOUNT, lr=0.001, tau=0.001, update_every=4)
   # logging
    logging_list = []

### QLEARNING LOOP
# initialize environment
environment = RaceEnv()
environment.init_render()
render = False
random_start = False # Only start at random position after n episodes

# Q-learning loop
for episode in range(EPISODES):
    print(f"Episode: {episode}, epsilon: {round(epsilon,5)}")
    
    if (episode % SHOW_EVERY == 0):#only plot every n episodes after initial episodes are over
        if episode >= START_SHOWING_FROM:
            render = True
    random_start = episode >= START_RANDOM_STARTING_FROM
    done = False
    #checkpoint_reached = False
    steps = 0
    # logging
    qs_episode = [] #Q values
    rs_episode = [] # rewards
    tdes_episode = [] # TD error
    # get initial state
    current_state = environment.reset(random_start = random_start)

    while not done:
        # Get action: exploit or explore
        action = q_agent.act(current_state, epsilon)
        # perform
        new_state, reward, done, checkpoint_reached = environment.step(actions[action])

        # render current state
        if render:
            environment.render()

        # Get !s from loacl and target networks, every n runs train them
        max_future_q, current_q, new_q, td_error = q_agent.step(current_state, action, reward, new_state, done)

        print(f"Tropy reached! Q: {new_q}, R: {reward}") if environment.win_condition else None
      
        # Update current state for new loop
        current_state = new_state

        # log
        tdes_episode.append(td_error)
        qs_episode.append(new_q)
        rs_episode.append(reward)
        steps += 1

    # Add a later stage, add pulse to epsilon and restart decaying
    if episode == EPSILON_PULSE_AT:
        epsilon += EPSILON_PULSE       
        START_EPSILON_DECAYING = EPSILON_PULSE_AT
        END_EPSILON_DECAYING = EPSILON_PULSE_AT + (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    print("Steps: ", steps)
    render = False
    episode_log = [steps, epsilon, np.sum(qs_episode), np.sum(rs_episode), np.sum(tdes_episode), np.max(qs_episode), np.min(qs_episode),np.quantile(qs_episode, 0.9), np.max(rs_episode), np.min(rs_episode),np.quantile(rs_episode, 0.9), 
                   environment.car.rect.center[0], environment.car.rect.center[1]]
    # episode_log = [steps, epsilon, np.sum(qs_episode), np.sum(rs_episode), np.sum(tdes_episode), np.quantile(tdes_episode,0.1),np.quantile(tdes_episode,0.9), np.max(qs_episode), np.min(qs_episode),np.quantile(qs_episode, 0.9), np.max(rs_episode), np.min(rs_episode),np.quantile(rs_episode, 0.9), 
    #                environment.car.rect.center[0], environment.car.rect.center[1]]
    logging_list.append(episode_log)
    # print(logging_list)
    # print(episode_log)

    # every 25 episodes, save models + results
    if episode % 10 == 0 and SAVE:
        # save models
        torch.save(q_agent.qnetwork_local.state_dict(), f"results/{FOLDER_NAME}/qnetwork_local.pth")
        torch.save(q_agent.qnetwork_target.state_dict(), f"results/{FOLDER_NAME}/qnetwork_target.pth")
        pd.DataFrame(logging_list, columns=logging_cols).to_feather(f"results/{FOLDER_NAME}/logging.feather")
# final save
if SAVE:
        torch.save(q_agent.qnetwork_local.state_dict(), f"results/{FOLDER_NAME}/qnetwork_local.pth")
        torch.save(q_agent.qnetwork_target.state_dict(), f"results/{FOLDER_NAME}/qnetwork_target.pth")
        pd.DataFrame(logging_list, columns=logging_cols).to_feather(f"results/{FOLDER_NAME}/logging.feather")

pygame.quit()
