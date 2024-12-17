IMAGEPATH = 'Race_Game/images/'
WINDOW_WIDTH, WINDOW_HEIGHT = 1020, 770#1024, 768
### Reward Parameters
# Win/loss conditions (overwrite all else)
MAX_REWARD = 80 #trophy reached
MAX_PENALTY = -80 # collision
# NO-GO zones to wall + pads
BUFFER_RATIO = 2 #Safety distance to walls + obstacles, discrete drop in reward (buffered car = car size * BUFFER_RATIO)
BUFFER_PENALTY = -10 #if exceeding safety buffer to walls + obstacles
# Continous distance measures
DISTANCE_PENALTY = -18 #if too close to walls or obstacles
DISTANCE_CHECKPOINT_REWARD = 10 #the closer to checkpoint the better
# Add reward for reaching checkpoints
CHECKPOINT_REWARD = MAX_REWARD/6 #checkpoint reached

# Car Parameters
MAX_SPEED = 8.1
MIN_SPEED = -7.9
ACCELERATION = 4
TURN_ACCELERATION = 15 #in degrees
VIEW = 80 #viewing distance of driver in 8 whisker directions
    