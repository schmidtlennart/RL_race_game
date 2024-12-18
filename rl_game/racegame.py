import pygame
import numpy as np
from pygame.locals import *
import gym
from rl_game.game_elements import CarSprite, PadSprite, CheckpointSprite, Trophy
from rl_game.game_helpers import scale_rect
from rl_game.game_config import *
class RaceEnv(gym.Env):
    # environment class based off of gym.Env
    # Screen definition (x,y): Top left: (0,0), Bottom right: (WINDOW_WIDTH,WINDOW_HEIGHT)
    def __init__(self):
        # Parameters that might be changed from outside
        # whether to start at random position
        self.random_start = False
        self.initialize_environment()

    def initialize_environment(self):
        # initialize pygame
        pygame.init()
        # set fps timer
        self.clock = pygame.time.Clock()
        # Initialize variables
        self.win_condition = None 
        self.reward_dict = {key: np.nan for key in INDIVIDUAL_REWARDS}
        # create obstacles (x,y,width)
        walls_list = [((-11, WINDOW_HEIGHT/2), 25, WINDOW_HEIGHT), #left
                ((WINDOW_WIDTH/2, -11), WINDOW_WIDTH, 25), #top
                ((WINDOW_WIDTH+11, WINDOW_HEIGHT/2), 25, WINDOW_HEIGHT), #right
                ((WINDOW_WIDTH/2, WINDOW_HEIGHT+11), WINDOW_WIDTH, 25)]#bottom
        pads_list = [((50, 10), 400), ((740, 10), 800), ((350,160),900), ((150,310),400), ((800,310),500), ((650,460),900), ((50,610),800),((850,610),400), ((500,760),1100)]
        self.pads = [PadSprite(position, width) for position, width in pads_list] + [PadSprite(position, width, height) for position, width, height in walls_list]
        self.pad_group = pygame.sprite.Group(*self.pads)
        # Explicit Guidance: define y-checkpoints on the way to trophy (if it passes through y of obstacles)
        checkpoints_list = [(550,610),(100,460), (450,310), (900,160)]
        self.checkpoints = [CheckpointSprite(checkpoint) for checkpoint in checkpoints_list]
        self.checkpoint_group = pygame.sprite.Group(self.checkpoints)
        self.checkpoint_counter = 0
        self.checkpoint_reward = 0
        # create trophy
        self.trophy = Trophy((298,20))
        self.trophy_group = pygame.sprite.Group(self.trophy)#only needed for collision calculation
        # create car
        start_position = (50, 680)
        if self.random_start:
            #add delta
            start_position = (start_position[0] + np.random.randint(0,900),start_position[1] + np.random.randint(-20,20))
        self.car = CarSprite(IMAGEPATH+'car.png', start_position)
        self.car_group = pygame.sprite.Group(self.car)#only needed for collision calculation
        # calculate first whiskers
        self.calculate_whiskers()       

    def init_render(self):      
        # render obstacles, car, trophy 
        self.pad_group = pygame.sprite.RenderPlain(self.pad_group)
        self.car_group = pygame.sprite.RenderPlain(self.car_group)
        self.trophy_group = pygame.sprite.RenderPlain(self.trophy_group)
        self.checkpoint_group = pygame.sprite.RenderPlain(self.checkpoint_group)
        # set up font
        self.font = pygame.font.Font(None, 20)
        self.screenmessage = self.font.render('', True, (255,0,0))
        # set up game window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


    def render(self):
        self.screen.fill((0,0,0))# empty screen        
        self.pad_group.update(self.collisions)#change pad image if hit
        self.car_group.draw(self.screen)#draw car
        for w in self.whiskers:
            pygame.draw.line(self.screen, "white", *w)
        self.pad_group.draw(self.screen)
        self.trophy_group.draw(self.screen)
        self.checkpoint_group.draw(self.screen)
        self.screen.blit(self.font.render(self.screenmessage, True, (0,255,0)), (580, 730))
        pygame.display.flip()

    def reset(self, random_start = False):
        # reset the environment to initial state
        self.random_start = random_start #set random start point
        #return observation
        self.initialize_environment()
        self.screenmessage = ''
        # give out current state
        state = self.get_state()
        return state

    def get_state(self):
        # return obsrevations etc. for training
        # for speed, round because floats caused problems in discretization
        state = np.concatenate([self.distances, np.array([self.car.direction, np.round(self.car.speed)])])
        return state       

    def check_loss_win(self):
        ### CHECK WIN OR LOSS STATES
        # reward is MAX_PENALTY if collision with pads or walls
        self.collisions = pygame.sprite.groupcollide(self.car_group, self.pad_group, False, False, collided = None)
        if self.collisions != {}:
            self.win_condition = False

        # reward is MAX_REWARD if trophy is reached
        trophy_collision = pygame.sprite.groupcollide(self.car_group, self.trophy_group, False, True)
        if trophy_collision != {}:
            self.win_condition = True

        # 1. WIN or LOSS: Save reward directly if game lost or won        
        if self.win_condition is not None:
            win_loss_reward = [MAX_PENALTY,MAX_REWARD][int(self.win_condition)]
            # reset reward dict
            self.reward_dict = {key: np.nan for key in self.reward_dict}           
            self.reward_dict["Win/Loss"] = win_loss_reward
            self.reward = win_loss_reward

    def calculate_reward(self):
        #### CALCULATE REWARD
        # reset reward dict
        self.reward_dict = {key: np.nan for key in self.reward_dict}
        
        ### 1. CHECK WIN OR LOSS STATES
        # if won or lost, abort
        self.check_loss_win()
        if self.win_condition is not None:
            return

        ### 2. PAD BUFFER if too close to pads and walls using buffer around car
        # buffers around car to use as buffer around pads and walls
        buffer_penalties = []
        for buffer_ratio in BUFFER_RATIOS:
            buffered_car = scale_rect(self.car.rect, buffer_ratio)
            buffer_collisions = [buffered_car.colliderect(pad.rect) for pad in self.pads]        
            if np.any(buffer_collisions): 
                buffer_penalties.append(BUFFER_PENALTY/(buffer_ratio**1.5)) #penalty decreases with increasing buffer ratio
            
        self.reward_dict["Wall/Pad Buffer"] = sum(buffer_penalties)

        ### 3. DISTANCE to both pads and walls
        # distance_penalty = 1-np.array(self.distances)/(VIEW*1.1) #normalize by maximum view distance to [0,1]
        # distance_penalty = np.sum(distance_penalty)*DISTANCE_PENALTY #alternatively: max = only respect closest object
        # reward_list.append(distance_penalty)
 
        # ### distance to trophy
        # distance_trophy = np.array(self.car.rect.center) - np.array(self.trophy.rect.center)
        # # normalize by screen width, height, i.e. between 0 and 1 * BUFFER_PENALTY
        # distance_trophy = distance_trophy * (0.2,0.8) #weighting x less than y
        # distance_trophy = round(1/(np.sum(distance_trophy / np.array([WINDOW_WIDTH, WINDOW_HEIGHT]))),1)
        # #reward.append(distance_trophy)

        ### 4. CHECKPOINT REACHED
        # if checkpoint n is reached, overwrite Q directly with reward and add constant to reward from there on
        self.checkpoint_reached = False
        if self.checkpoints[self.checkpoint_counter].rect.collidepoint(self.car.rect.center):
            self.checkpoint_reward = CHECKPOINT_REWARD*(self.checkpoint_counter+1)#needs to be 1-indexed
            # remove checkpoint from to make sure each checkpoint is only counted once
            self.checkpoints[self.checkpoint_counter].rect.center = (-100,-100)
            if (self.checkpoint_counter < len(self.checkpoints)-1):#as long as not in final zone, next checkpoint has to be reached
                    self.checkpoint_counter += 1
            # set reward to checkpoint reward and 
            self.reward = self.checkpoint_reward
            self.reward_dict["Checkpoint Level"] = self.checkpoint_reward
            self.checkpoint_reached = True
            self.screenmessage = f"Checkpoint reached! Reward: {round(self.reward,1)}"
            return
        # if not reached simply continue adding constant for having made above checkpoint n    
        self.reward_dict["Checkpoint Level"] = self.checkpoint_reward
        
        ### 5. DISTANCE TO NEXT CHECKPOINT 
        distance_checkpoint = np.array(self.car.rect.center) - np.array(self.checkpoints[self.checkpoint_counter].rect.center)
        # normalize by screen width, height, i.e. between 0 and 1 
        distance_checkpoint = (np.abs(distance_checkpoint) / np.array([WINDOW_WIDTH, WINDOW_HEIGHT]))
        # subtract from 1
        distance_checkpoint = (1-np.sum(distance_checkpoint))*DISTANCE_CHECKPOINT_REWARD
        self.reward_dict["Distance to Checkpoint"] = distance_checkpoint
        
        ### sum up all penalties
        self.reward = np.nansum(list(self.reward_dict.copy().values()))
        
        # update stats in screen message
        self.screenmessage = f"Reward: {round(self.reward,1)} Buffer: {round(self.reward_dict['Wall/Pad Buffer'],1)}"# Wall Distance Penalty: {round(distance_penalty,1)} Checkpoint Distance: {round(distance_checkpoint,1)}"

    def compute_reward_map(self):
        reward_map = np.full((WINDOW_WIDTH, WINDOW_HEIGHT, len(self.reward_dict)+1), np.nan, dtype=np.float32)
        
        # Precompute pad collision masks
        pad_masks = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT), dtype=bool)
        for pad in self.pads:
            # clip to window size
            left = max(0, pad.rect.left)
            right = min(WINDOW_WIDTH, pad.rect.right)
            top = max(0, pad.rect.top)
            bottom = min(WINDOW_HEIGHT, pad.rect.bottom)
            pad_masks[left:right, top:bottom] = True

        # optional: rotate car as buffers change
        for i in range(6):#6*15*90 degrees
            self.car.update([0,1])

        # Compute the reward map
        for y in range(WINDOW_HEIGHT-1, -1,-1): #-1
            if y % 50 == 0:
                print(y)
            for x in range(WINDOW_WIDTH):
                #self.initialize_environment()
                self.win_condition = None
                self.car.rect.center = (x, y)
                
                # Skip if center inside any of the pads
                if pad_masks[x, y]:
                    continue
                # for faster computation, check win/loss first and skip if won/lost
                self.check_loss_win() 
                if self.win_condition is None:              
                    self.calculate_whiskers()#expensive
                    self.calculate_reward()  # Get total + individual rewards
                reward_map[x, y, :] = [self.reward] + list(self.reward_dict.values())

        return reward_map

    def calculate_whiskers(self):
        # whiskers to "see" surrounding objects if whiskers collide with them at a given distance
        view = VIEW
        view_c = 0.75*view #diagonal ones
        self.whiskers = [(self.car.rect.center, (self.car.rect.center[0], self.car.rect.center[1]-view)),#top
                (self.car.rect.center, (self.car.rect.center[0]+view, self.car.rect.center[1])),#right
                (self.car.rect.center, (self.car.rect.center[0], self.car.rect.center[1]+view)),#bottom
                (self.car.rect.center, (self.car.rect.center[0]-view, self.car.rect.center[1])),#left  
                (self.car.rect.center, (self.car.rect.center[0]+view_c, self.car.rect.center[1]-view_c)),#top right
                (self.car.rect.center, (self.car.rect.center[0]+view_c, self.car.rect.center[1]+view_c)), #bottom right
                (self.car.rect.center, (self.car.rect.center[0]-view_c, self.car.rect.center[1]+view_c)),#bottom left
                (self.car.rect.center, (self.car.rect.center[0]-view_c, self.car.rect.center[1]-view_c))]#top left
        
        # get collision points of whiskers with other objects
        w_collisions = []
        w_distances = []
        for w in self.whiskers:
            
            ### PAD COLLISIONS
            # Get intersection point of whisker line with each pad
            w_collisions_w = [pad.rect.clipline(*w) for pad in self.pads]
            
            ### DISTANCE TO COLLISION POINTS
            # clean up: drop empty tuples, keep only tuple (x,y) of actual collision points
            w_collisions_w = [wcp[0] for wcp in w_collisions_w if wcp != ()]
            # calculate distance to collision points
            if w_collisions_w != []:
                w_distances_pads = np.array([np.linalg.norm(self.car.rect.center - np.array(wcp)) for wcp in w_collisions_w])
                w_argmin = w_distances_pads.argmin()#get closest collision
                w_collisions.append(w_collisions_w[w_argmin])            
            else: # no collision between whisker and pad/wall
                w_distances_pads = np.array([view*1.1]) # set to beyond max view
            w_distances.append(w_distances_pads.min())
        self.whisker_collisions = w_collisions
        self.distances = w_distances


    def step(self, action):
        # perform one step in the game logic
        # move car
        self.car.update(action)
        # calculate whiskers
        self.calculate_whiskers()
        # get state
        state = self.get_state()
        # check collision, calc reward
        self.calculate_reward()

        # print win/loss message
        done = self.win_condition is not None
        if done:
            self.screenmessage = ['You messed up. Press Space to retry', 'Finished! Press Space to retry'][int(self.win_condition)] + f" - Reward: {round(self.reward,1)}"
            self.car.MAX_SPEED = 0
            self.car.MIN_SPEED = 0
            if not self.win_condition:#if lost, show collision image
                self.car.image = pygame.image.load(IMAGEPATH+'collision.png')
        return state, self.reward, done, self.checkpoint_reached
    
    def pressed_to_action(self):
        # translate keyboard keys to exit/restart or game action    
        get_event = pygame.event.get()
        keytouple = pygame.key.get_pressed()
        # end game when window is closed or escape key is pressed
        for event in get_event:
            if event.type == pygame.QUIT or keytouple[pygame.K_ESCAPE]==1:
                pygame.quit()
            # reset the game when Failed or Won and space bar is pressed
            if (self.win_condition is not None) and keytouple[pygame.K_SPACE] == 1:
                _ = self.reset()
        action_turn = 0.
        action_acc = 0.
        if keytouple[pygame.K_UP] == 1:  # forward
            action_acc += 1
        if keytouple[pygame.K_DOWN] == 1:  # back
            action_acc -= 1
        if keytouple[pygame.K_LEFT] == 1:  # left
            action_turn += 1
        if keytouple[pygame.K_RIGHT] == 1:  # right
            action_turn -= 1
        return np.array([action_acc, action_turn])