import numpy as np
import pygame
from rl_game.racegame import RaceEnv, WINDOW_HEIGHT, WINDOW_WIDTH
import matplotlib.pyplot as plt


environment = RaceEnv()

environment.init_render()
action = environment.pressed_to_action()
# initial step
new_state, reward, done = environment.step(action)
initial_car_position = environment.car.rect.center

reward_map = environment.compute_reward_map()
pygame.quit()


# to float32
reward_map = np.array(reward_map, dtype=np.float32)
reward_map = reward_map.T

# save to numpy
np.save('results/reward_map.npy', reward_map)


reward_map = np.load('results/reward_map.npy')
#np.unique(reward_map, return_counts=True)

# Create a 2x1 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))

# Plot the reward map in the first subplot
im = ax1.imshow(reward_map, interpolation="none")
fig.colorbar(im, ax=ax1)
# Plot initial car position as red dot
ax1.scatter(initial_car_position[0], initial_car_position[1], c='red')
# Get pads and overlay as gray rectangles from .left, .right, .top, .bottom
for pad in environment.pads:
    ax1.add_patch(plt.Rectangle((pad.rect.left, pad.rect.top), pad.rect.width, pad.rect.height, fill=True, edgecolor='darkgray', color="white", lw=1))
# Set xlims to window width, ylims to window height
ax1.set_xlim(0, WINDOW_WIDTH)
ax1.set_ylim(WINDOW_HEIGHT, 0)
ax1.set_title('Reward Map')

# Plot the histogram of reward values in the second subplot
ax2.hist(reward_map[~np.isnan(reward_map)].flatten(), bins=50, color='green', alpha=0.4, edgecolor='black')
ax2.set_title('Histogram of Reward Values')
ax2.set_xlabel('Reward Value')
ax2.set_ylabel('Frequency')

# Save to PNG
plt.tight_layout()
plt.savefig('images/reward_map.png', dpi=300)
plt.close()


reward_map.min()
reward_map.max()
# min_index = np.argmin(m)

# # Convert the flattened index to coordinates in the 2D array
# min_coords = np.unravel_index(min_index, m.shape)
