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
    
def get_wall_collision(wall, whisker):
    # Unpack the points
    (x1, y1), (x2, y2) = wall
    (x3, y3), (x4, y4) = whisker
    # Calculate line coefficients
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
    ix, iy = np.linalg.solve(A, B)
    # Check if the intersection point is within the x,y bounds of the whisker i.e. on it
    if min(x3, x4) <= ix <= max(x3, x4) and min(y3, y4) <= iy <= max(y3, y4):
        return ((ix, iy),())
    else:
        return ()  # Intersection point is not within the bounds of both line segments
