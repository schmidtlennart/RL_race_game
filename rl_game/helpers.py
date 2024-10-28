import numpy as np

### QLEARNING HELPERS
def get_discrete_state(observations, in_bins):
    # map current observation to bins of the Q-table
    out_bins = []
    for obs, bins in zip(observations, in_bins):
        out_bin = np.digitize(obs, bins)-1 # -1 because bins are 1-indexed
        out_bins.append(out_bin)
    return tuple(out_bins)

# find point of intersection between two lines
def find_intersection(line1, line2):
    # Unpack the points
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    # Calculate the coefficients of the lines
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1

    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3

    # Set up the system of equations
    A = np.array([[A1, B1], [A2, B2]])
    B = np.array([C1, C2])

    # Check if the lines are parallel
    if np.linalg.det(A) == 0:
        return ()  # Lines are parallel and do not intersect

    # Solve the system of equations
    intersection = np.linalg.solve(A, B)
    return intersection