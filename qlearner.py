import sys
import numpy as np
import pygame
from rl_game.racegame import RaceEnv
from rl_game.helpers import get_discrete_state, calc_bins, PygameRecord

# Check if "record" argument is present in sys.argv
RECORD = "record" in sys.argv

# Initialize environment and other variables
environment = RaceEnv()
environment.init_render()
all_bins = calc_bins()
q_table = np.load("results/q_table.npy")  # Load Q-table
actions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # Define actions

q_table.shape

# run 2d-PCA on q_table and visualize how the agent moves
# 1. PCA on q_table
# 2. plot the first two components
# 3. plot the trajectory of the agent

# 1. PCA on q_table
# first reshape to 2D along actions
num_samples = np.prod(q_table.shape[:-1])  # Calculate the number of samples
num_actions = q_table.shape[-1]  # The last dimension represents the actions
q_table_reshaped = q_table.reshape(num_samples, num_actions)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
q_table_pca = pca.fit_transform(q_table_reshaped)

q_table_reshaped.shape
q_table_pca.shape

def plot_2d_qtable(q_table_pca):
    import matplotlib.pyplot as plt
    plt.scatter(q_table_pca[:, 0], q_table_pca[:, 1])
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("2D PCA of Q-table")
    plt.savefig("images/q_table_pca.png")

plot_2d_qtable(q_table_pca)

# Start position
discrete_state = get_discrete_state(environment.reset(random_start=True), all_bins)
epsilon = 0.0005
done = False
n_frames = 2000  # Number of frames to record

# Conditionally initialize the recorder
if RECORD:
    recorder = PygameRecord("gifs/recording.gif", 30)
step_counter = 0

try:
    while not done:
        # Set game speed to 30 fps
        environment.clock.tick(30)
        
        # Still needs some epsilon otherwise gets stuck in minima
        if np.random.random() > epsilon:
            # Exploit: Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Explore: Get random action
            action = np.random.randint(0, len(actions))
        action = actions[action]
        
        # Calculate one step
        new_state, _, done, _ = environment.step(action)
        discrete_state = get_discrete_state(new_state, all_bins)
        
        # Render current state
        environment.render()
        
        # count step
        step_counter += 1
        # Conditionally record the frame
        if RECORD:
            recorder.add_frame()
            n_frames -= 1
            if n_frames == 0:
                break
finally:
    print("n Steps: ", step_counter)
    # Conditionally save the recording and quit pygame
    if RECORD:
        recorder.save()
    pygame.quit()


        # overwrite if manual key pressed
        #action = environment.pressed_to_action()

    ### Let Qlearner start randomly and just let it run


    # class Qlearner():
    #     def __init__(self, environment):
    #         self.environment = environment
    #         #self.q_table = self.create_q_table()
    #         #self.actions = self.environment.ACTIONS
    #         #self.all_bins = self.environment.calc_bins()
    #         self.epsilon = 1
    #         self.EPISODES = 10000
    #         self.START_SHOWING_FROM = 1000
    #         self.START_RANDOM_STARTING_FROM = 1000
    #         self.LEARNING_RATE = 0.1
    #         self.DISCOUNT = 0.95
    #         self.SHOW_EVERY = 1000
    #         self.render = False
    #         self.random_start = False
        
    #     def train(self):
    #         pass

    #     def run(self):
    #         # 
