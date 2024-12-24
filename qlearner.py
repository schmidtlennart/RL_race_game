
import pygame
from rl_game.racegame import RaceEnv
from rl_game.game_config import VIEW, TURN_ACCELERATION, ACCELERATION, MIN_SPEED, MAX_SPEED
import numpy as np
from rl_game.game_config import VIEW
from rl_game.helpers import get_discrete_state, calc_bins

all_bins = calc_bins()

q_table = np.load("results/q_table.npy")

########################
    environment = RaceEnv()
    environment.init_render()
    # start position
    discrete_state = get_discrete_state(environment.reset(random_start = True), all_bins)


    run = True

    while run:

        # set game speed to 30 fps
        environment.clock.tick(30)
        # ─── CONTROLS ───────────────────────────────────────────────────────────────────
        # get pressed keys, generate action
        action = np.argmax(q_table[discrete_state])
        # calculate one step
        new_state, _,_,_ = environment.step(action)
        ########
        discrete_state = get_discrete_state(new_state, all_bins)    
        # render current state
        environment.render()

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
