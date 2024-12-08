import numpy as np

## QLEARNING HELPERS
def get_discrete_state(observations, in_bins):
    # map current observation to bins of the Q-table
    out_bins = []
    len(observations)
    len(in_bins)
    for obs, bins in zip(observations, in_bins):
        bin_index = np.digitize(obs, bins, right=True)
        
        # Ensure values below the first bin end up in the first bin if bigger than the first bin
        if bin_index == 0:
            bin_index = 1
        
        # Ensure values above the last bin end up in the last bin
        elif bin_index == len(bins):
            bin_index = len(bins) - 1
        out_bins.append(bin_index-1)# Adjust index to be zero-based
    return tuple(out_bins)
