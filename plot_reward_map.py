import numpy as np
import pygame
from rl_game.racegame import RaceEnv, WINDOW_HEIGHT, WINDOW_WIDTH
import matplotlib.pyplot as plt


environment = RaceEnv()

environment.init_render()
action = environment.pressed_to_action()
# calculate one step
new_state, reward, done = environment.step(action)
initial_car_position = environment.car.position
map = environment.plot_reward_map()
# to float32
map = np.array(map, dtype=np.float32)
map = map.T


plt.imshow(map, interpolation="none", norm="log")
plt.colorbar()
# plot initial car position as red dot
plt.scatter(initial_car_position[0], initial_car_position[1], c='red')
# get pads and overlay as gray rectangles from .left, .right, .top, .bottom
for pad in environment.pads:
    plt.gca().add_patch(plt.Rectangle((pad.rect.left, pad.rect.top), pad.rect.width, pad.rect.height, fill=False, edgecolor='darkgray', lw=1))
# set xlims to window width, ylims to window height
plt.xlim(0, WINDOW_WIDTH)
plt.ylim(WINDOW_HEIGHT,0)
# save to png
plt.savefig('images/reward_map_2.png', dpi=300)
# save to numpy
np.save('results/reward_map_2.npy', map)
plt.close()

pygame.quit()

map = np.load('results/reward_map_2.npy')
min_index = np.argmin(m)

# Convert the flattened index to coordinates in the 2D array
min_coords = np.unravel_index(min_index, m.shape)
