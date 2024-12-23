
import pygame
from rl_game.racegame import RaceEnv
from rl_game.game_config import VIEW, TURN_ACCELERATION, ACCELERATION, MIN_SPEED, MAX_SPEED
import numpy as np
from rl_game.game_config import VIEW
from rl_game.helpers import get_discrete_state, calc_bins, PygameRecord

all_bins = calc_bins()

q_table = np.load("results/q_table.npy")

actions = [(1,0), (-1,0), (0,1), (0,-1)]

n_frames=500

with PygameRecord("gifs/recording.gif", 30) as recorder: 
    ########################
    environment = RaceEnv()
    environment.init_render()
    # start position
    discrete_state = get_discrete_state(environment.reset(random_start = True), all_bins)

    epsilon=0.1

    done = False
    while not done:

        # set game speed to 30 fps
        environment.clock.tick(30)
        #still needs some epsilon otherwise gets stuck in minima
        if np.random.random() > epsilon:
            # Exploit: Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Explore: Get random action
            action = np.random.randint(0, len(actions))
        action = actions[action]
        # overwrite if manual key pressed
        #action = environment.pressed_to_action()
        #print(action)
        # calculate one step
        new_state, _,done,_ = environment.step(action)
        ########
        discrete_state = get_discrete_state(new_state, all_bins)    
        # render current state
        environment.render()
        # record
        recorder.add_frame()
        n_frames -=1
        if n_frames == 0:
            break
    recorder.save()
    pygame.quit()


    ### Let Qlearner start randomly and just let it run


    # class Qlearner():
    #     def __init__(self, environment):
    #         self.environment = environment
    #         #self.q_table = self.create_q_table()
    #         #self.actions = self.environment.ACTIONS
    #         #self.all_bins = self.environment.calc_bins()
    #         self.epsilon = 1
    #         self.EPISODES = 10000
    #         self.START_SHOWING_FROM = 1000
    #         self.START_RANDOM_STARTING_FROM = 1000
    #         self.LEARNING_RATE = 0.1
    #         self.DISCOUNT = 0.95
    #         self.SHOW_EVERY = 1000
    #         self.render = False
    #         self.random_start = False
        
    #     def train(self):
    #         pass

    #     def run(self):
    #         # 
