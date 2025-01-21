import numpy as np
### Helper functions: How to put into separate file?
# Function to scale the car rectangle by ratio for near miss calculation
def scale_rect(rect, ratio):
    new_width = rect.width * ratio
    new_height = rect.height * ratio
    new_rect = rect.copy()
    new_rect.width = new_width
    new_rect.height = new_height
    new_rect.center = rect.center  # Keep the center the same
    return new_rect

def degree_to_sin_cos(d):
    r = d * np.pi / 180.#convert to radian (required for cos+sin)
    return np.array([np.cos(r), np.sin(r)])#represent degree as [cos,sin]