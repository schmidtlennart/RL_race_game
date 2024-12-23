import pygame
from PIL import Image
import numpy as np
import os
from rl_game.game_config import VIEW, TURN_ACCELERATION, ACCELERATION, MIN_SPEED, MAX_SPEED

## QLEARNING HELPERS
def get_discrete_state(observations, in_bins):
    # map current observation to bins of the Q-table
    out_bins = []
    for obs, bins in zip(observations, in_bins):
        # clip observations to the range of the bins
        clipped_obs = np.clip(obs, bins[0], bins[-1])
        bin_index = np.digitize(clipped_obs, bins, right=True)
        
        # # Ensure values below the first bin end up in the first bin if bigger than the first bin
        # if bin_index == 0:
        #     bin_index = 1
        
        # # Ensure values above the last bin end up in the last bin
        # elif bin_index == len(bins):
        #     bin_index = len(bins) - 1
        out_bins.append(bin_index)# Adjust index to be zero-based
    return tuple(out_bins)

### STATES & ACTIONS: create bins of continous states for the Q-table
def calc_bins():
    # get_discrete_states uses np.digitize that checks x <= bins[i]. so drop the first bin value
    # distances of 8 whiskers, add separate bin for when whisker sees nothing (set to VIEW*1.1)
    distances_bins = [np.concat([np.linspace(15, VIEW, 4)[1:],np.array([VIEW*1.1])])]*8 #= 3 bins + 1 bin for "far away"
    # for direciton and speed, needs to be designed such that each action ends up in a new bin
    # direction: [0, 359]
    # still one bin too much: 0 gets its own discrete bin
    direction_bins = np.linspace(0, 360, int(360/TURN_ACCELERATION+1))
    # speed: [-7.9, 8.1] in ACCELERATION steps
    # BUG: Float rounding problems - currently as integers
    #speed_bins = np.linspace(MIN_SPEED, MAX_SPEED, int((abs(MIN_SPEED)+MAX_SPEED)/ACCELERATION+1))
    speed_bins = np.linspace(-8, 8, int(16/ACCELERATION+1))
    all_bins = distances_bins + [direction_bins] + [speed_bins]
    return all_bins

class PygameRecord:
    def __init__(self, filename: str, fps: int):
        self.fps = fps
        # check if filename exists, if so add _1, _2, _3, etc.
        base, ext = filename.rsplit('.', 1)
        counter = 1
        new_filename = f"{base}_{counter}.{ext}"
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{base}_{counter}.{ext}"
        self.filename = new_filename
        self.frames = []

    def add_frame(self):
        curr_surface = pygame.display.get_surface()
        x3 = pygame.surfarray.array3d(curr_surface)
        x3 = np.moveaxis(x3, 0, 1)
        array = Image.fromarray(np.uint8(x3))
        self.frames.append(array)

    def save(self):
        self.frames[0].save(
            self.filename,
            save_all=True,
            optimize=False,
            append_images=self.frames[1:],
            loop=0,
            duration=int(1000 / self.fps),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred: {exc_value}")
        self.save()
        # Return False if you want exceptions to propagate, True to suppress them
        return False