IMAGEPATH = 'Race_Game/images/'
WINDOW_WIDTH, WINDOW_HEIGHT = 1020, 770#1024, 768
### Reward Parameters
# list of all individual reward maps that are summed for final reward value
INDIVIDUAL_REWARDS = ["Win/Loss", "Wall/Pad Buffer", "Checkpoint Level", "Distance to Checkpoint"]#"Distance to Wall/Pad", 
# Win/loss conditions (overwrite all else)
MAX_REWARD = 80 #trophy reached
MAX_PENALTY = -80 # collision
# NO-GO zones to wall + pads
BUFFER_RATIOS = [1.2, 1.4, 1.6, 1,8, 2, 2.2] #Safety distance to walls + obstacles, discrete drop in reward (buffered car = car size * BUFFER_RATIO)
BUFFER_PENALTY = -20 #if exceeding safety buffer to walls + obstacles
# Continous distance measures
# DISTANCE_PENALTY = -18 #if too close to walls or obstacles
DISTANCE_CHECKPOINT_REWARD = 7 #the closer to checkpoint the better
# Add reward for reaching checkpoints
CHECKPOINT_REWARD = MAX_REWARD/8 #checkpoint reached

# Car Parameters
# for qtable: [-7.9, 8.1], turn_acc: 15
MAX_SPEED = 8 
MIN_SPEED = -3
ACCELERATION = 4
TURN_ACCELERATION = 10 #in degrees
VIEW = 80 #viewing distance of driver in 8 whisker directions
