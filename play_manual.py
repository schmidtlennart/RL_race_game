import pygame
from rl_game.racegame import RaceEnv
from rl_game.game_config import VIEW, TURN_ACCELERATION, ACCELERATION, MIN_SPEED, MAX_SPEED

########################
# debug binning
import numpy as np
from rl_game.helpers import get_discrete_state, calc_bins

all_bins = calc_bins()

########################

environment = RaceEnv()
environment.init_render()

run = True
steps=0
while run:
    # smoother controls
    #environment.car.ACCELERATION = 0.5
    #environment.car.TURN_ACCELERATION = 5
    steps+=1
    # set game speed to 30 fps
    environment.clock.tick(30)
    # ─── CONTROLS ───────────────────────────────────────────────────────────────────
    # get pressed keys, generate action
    action = environment.pressed_to_action()
    # calculate one step
    new_state, _,_,_ = environment.step(action)
    ########
    bin_indices = get_discrete_state(new_state, all_bins)
    #print("OBS: ",new_state[-0])#-1:speed, -2:direction, 0:distance top
    #print("BIN: ",bin_indices[-0])
    ########
    
    # render current state
    environment.render()

    if environment.win_condition:
        print(steps)

pygame.quit()


# import numpy as np
# from rl_game.game_config import VIEW
# from rl_game.helpers import get_discrete_state
# #distances_bins = [np.concat([np.linspace(0, VIEW, 4),np.array([VIEW*1.1])])]
# distances_bins = [np.concat([np.linspace(0, VIEW, 4)[1:],np.array([VIEW*1.1])])]*8 #0 is separate bin, not used
# distances_bins[0][0]
# # Example observations
# observations = [np.array([-10, 0, 0.1, 26, 27, 80, 88])]  # Example observations
# bin_index = get_discrete_state(observations, distances_bins)

# print("Observations:", observations)
# print("Bin Index:", bin_index)




# import numpy as np

# # Define your bins using np.linspace
# VIEW = 100  # Example value for VIEW
# n_bins = 4
# distances_bins = np.linspace(0, VIEW, n_bins)

# # Example observations
# observations = np.array([80, -10, 0, 120, 50])  # Example observations

# # Clip the observations to the range of your bins
# clipped_observations = np.clip(observations, distances_bins[0], distances_bins[-1])

# # Use np.digitize to bin the clipped observations
# bin_index = np.digitize(clipped_observations, distances_bins, right=True)

# print("Observations:", observations)
# print("Clipped Observations:", clipped_observations)
# print("Bin Index:", bin_index)