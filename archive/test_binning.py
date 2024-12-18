import numpy as np
from rl_game.game_config import VIEW, TURN_ACCELERATION, ACCELERATION, MIN_SPEED, MAX_SPEED
from rl_game.helpers import get_discrete_state, calc_bins

all_bins = calc_bins()
distances_bins = all_bins[0]
direction_bins = all_bins[-2]
speed_bins = all_bins[-1]

### TEST
# distance, direction, speed
obs_distance = np.array([-10, 0, 0.1, 26, 27, 80, 88])
obs_direction = np.array([359, 0,1, 15])
#obs_speed = np.array([-7.9, -3.9,  -3.8, 0.1, 0.2, 4, 4.1, 4.2,8.1], dtype=np.float64) #Float rounding prevents exact matching, so causes problems in binning, so as int in game
obs_speed = np.array([-8., -4.,  0.,  4.,  8.])
observations = [obs_distance, obs_direction, obs_speed]
test_bins = [distances_bins[0]] + [direction_bins] + [speed_bins]
bin_indices = get_discrete_state(observations, test_bins)

for i, obs in enumerate(observations):
    print("-------------------")
    print(f"Observation {i}: {obs}")
    print(f"Bin index {i}: {bin_indices[i]}")
    print(f"Bins: {test_bins[i]}")