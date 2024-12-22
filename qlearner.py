import pygame
import random



### Let Qlearner start randomly and just let it run


class Qlearner():
    def __init__(self, environment):
        self.environment = environment
        #self.q_table = self.create_q_table()
        #self.actions = self.environment.ACTIONS
        #self.all_bins = self.environment.calc_bins()
        self.epsilon = 1
        self.EPISODES = 10000
        self.START_SHOWING_FROM = 1000
        self.START_RANDOM_STARTING_FROM = 1000
        self.LEARNING_RATE = 0.1
        self.DISCOUNT = 0.95
        self.SHOW_EVERY = 1000
        self.render = False
        self.random_start = False
    
    def train(self):
        pass

    def run(self):
        # 