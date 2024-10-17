import numpy as np

### QLEARNING HELPERS
def get_discrete_state(observations, in_bins):
    # map current observation to bins of the Q-table
    out_bins = []
    for obs, bins in zip(observations, in_bins):
        out_bin = np.digitize(obs, bins)-1 # -1 because bins are 1-indexed
        out_bins.append(out_bin)
    return tuple(out_bins)