import numpy as np
import pygame
from rl_game.racegame import RaceEnv, WINDOW_HEIGHT, WINDOW_WIDTH
import matplotlib.pyplot as plt

INDIVIDUAL_REWARDS = ["Win/Loss", "Wall/Pad Buffer", "Distance to Wall/Pad", "Checkpoint Level", "Distance to Checkpoint"]

environment = RaceEnv()
initial_car_position = environment.car.rect.center

reward_map = environment.compute_reward_map()
pygame.quit()


# to float32
reward_map = np.array(reward_map, dtype=np.float32)
reward_map = reward_map.T

# save to numpy
np.save('results/reward_map.npy', reward_map)


reward_map = np.load('results/reward_map.npy')

# Create a subplot
n_plots = reward_map.shape[0] + 1
# automatically adjust the number of rows if two columns is fix
n_rows = int(np.ceil(n_plots/2))

fig, axs = plt.subplots(n_rows, 2, figsize=(20, 8*n_rows))
axs = axs.flatten()

# Plot the reward map in the first subplot
im = axs[0].imshow(reward_map[0,:,:], interpolation="none")
fig.colorbar(im, ax=axs[0])
# Plot initial car position as red dot
axs[0].scatter(initial_car_position[0], initial_car_position[1], c='red')
# Get pads and overlay as gray rectangles from .left, .right, .top, .bottom
for pad in environment.pads:
    axs[0].add_patch(plt.Rectangle((pad.rect.left, pad.rect.top), pad.rect.width, pad.rect.height, fill=True, edgecolor='darkgray', facecolor="lightgray", lw=1))
# Set xlims to window width, ylims to window height
axs[0].set_xlim(0, WINDOW_WIDTH)
axs[0].set_ylim(WINDOW_HEIGHT, 0)
axs[0].set_title('Reward Map')

# Plot the histogram of reward values in the second subplot
hist_data = reward_map[0,:,:].flatten()
axs[1].hist(hist_data, bins=50, color='green', alpha=0.4, edgecolor='black')
axs[1].set_xlabel('Reward Value')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Histogram of Reward Values')

for i, name in enumerate(INDIVIDUAL_REWARDS):
    index = i+1
    ax_index = index+1
    im = axs[ax_index].imshow(reward_map[index,:,:], interpolation="none")
    fig.colorbar(im, ax=axs[ax_index])
    axs[ax_index].set_title(INDIVIDUAL_REWARDS[i])
    axs[ax_index].set_xlim(0, WINDOW_WIDTH)
    axs[ax_index].set_ylim(WINDOW_HEIGHT, 0)



# Save to PNG
plt.tight_layout()
plt.savefig('images/reward_map.png', dpi=200)
plt.close()
