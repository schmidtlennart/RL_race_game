import numpy as np
import pygame
from rl_game.racegame import RaceEnv, WINDOW_HEIGHT, WINDOW_WIDTH
import matplotlib.pyplot as plt


environment = RaceEnv()

environment.car.rect.center = (280,24)
environment.calculate_reward()

environment.win_condition
environment.reward
environment.reward_list
