#initialize the screen
import pygame, math, time
import numpy as np
from pygame.locals import *
import gym

### ToDO
# remove car+trophy groups

WINDOW_WIDTH, WINDOW_HEIGHT = 1020, 770#1024, 768
IMAGEPATH = 'Race_Game/images/'

### Reward Parameters
MAX_REWARD = 20
CHECKPOINT_REWARD = MAX_REWARD/8
MAX_PENALTY = -20
BUFFER_RATIO = 2 #buffered car = car size * BUFFER_RATIO
BUFFER_PENALTY = -2 #if exceeding safety buffer to walls + obstacles

# Car Parameters
MAX_SPEED = 8
ACCELERATION = 0.5
TURN_ACCELERATION = 5
    

# Function to scale the car rectangle by ratio for near miss calculation
def scale_rect(rect, ratio):
    new_width = rect.width * ratio
    new_height = rect.height * ratio
    new_rect = rect.copy()
    new_rect.width = new_width
    new_rect.height = new_height
    new_rect.center = rect.center  # Keep the center the same
    return new_rect

class CarSprite(pygame.sprite.Sprite):
    
    def __init__(self, image, position):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load(image)
        self.position = position
        self.speed = 0
        self.direction = 320
        self.MAX_SPEED = MAX_SPEED

    def update(self, action):
        #SIMULATION
        # action[0]: acceleration -1:back, 1:forwards | action[1]: rotation, 1:left, -1:right
        # add acceleration to current speed
        self.speed += action[0]*ACCELERATION
        if abs(self.speed) > self.MAX_SPEED:
            self.speed = self.MAX_SPEED if self.speed > 0 else -self.MAX_SPEED
        # add change of direction to current direction
        self.direction += action[1]*TURN_ACCELERATION
        self.direction %= 360 #needs remapping to [0,359] because can take any value
        # calculate new position
        x, y = (self.position)
        rad = self.direction * math.pi / 180
        x += -self.speed*math.sin(rad)
        y += -self.speed*math.cos(rad)
        self.position = (x, y)
        # rotate image + rect accordingly
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect()
        self.rect.center = self.position
        # return obsrevations etc. for training
        state = (self.position[0], self.position[1], self.direction, self.speed)
        return state

class PadSprite(pygame.sprite.Sprite):
    # Dimensions: PadSprite((100,100))#x: -150,350, y: 113, 88 = +/-250 in x, +/- 13 in y = size 500, 25
    normal = pygame.image.load(IMAGEPATH+'race_pads.png')
    hit = pygame.image.load(IMAGEPATH+'collision.png')
    def __init__(self, position):
        super(PadSprite, self).__init__()
        self.rect = pygame.Rect(self.normal.get_rect())
        self.rect.center = position
    def update(self, hit_list):
        # if car collided with pad, change image to hit
        if self in hit_list:
            self.image = self.hit
        else:
            self.image = self.normal

class CheckpointSprite(pygame.sprite.Sprite):
    def __init__(self, position, width=150, height=25):
        super(CheckpointSprite, self).__init__()
        #self.rect = pygame.Rect(0, 0, width, height)
        #self.rect.center = position
        self.image = pygame.Surface((width, height))
        self.image.fill((255, 0, 0))  # Fill the pad with a color (red in this case)
        self.rect = self.image.get_rect()
        self.rect.center = position

class Trophy(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(IMAGEPATH+'trophy.png')
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = position
    def draw(self, screen):
        screen.blit(self.image, self.rect)

class RaceEnv(gym.Env):
    # environment class based off of gym.Env
    # Screen definition (x,y): Top left: (0,0), Bottom right: (1024,768)
    def __init__(self,env_config={}):
        # win/fail message in terminal
        self.message_printed = False
        # initialize pygame
        pygame.init()
        # set fps timer
        self.clock = pygame.time.Clock()
        # Initialize variables
        self.win_condition = None 
        # create obstacles
        pads_list = [(0,10), (600,10), (1100,10), (100,150), (600,150), (100,300), (800,300), (400,450), (900,450), (200,600), (900,600), (400,750), (800,750)]
        self.pads = [PadSprite(pad) for pad in pads_list]
        self.pad_group = pygame.sprite.Group(*self.pads)
        # define y-checkpoints (if it passes through y of obstacles)
        checkpoints_list = [(550,600),(50,450), (450,300), (950,150)]
        self.checkpoints = [CheckpointSprite(checkpoint) for checkpoint in checkpoints_list]
        self.checkpoint_group = pygame.sprite.Group(self.checkpoints)
        self.checkpoint_counter = 0
        self.checkpoint_reward = 0
        # create trophy
        self.trophy = Trophy((285,0))
        self.trophy_group = pygame.sprite.Group(self.trophy)#only needed for collision calculation
        # create car
        self.car = CarSprite(IMAGEPATH+'car.png', (10, 730))
        self.car_group = pygame.sprite.Group(self.car)#only needed for collision calculation       

    def init_render(self):      
        # render obstacles, car, trophy 
        self.pad_group = pygame.sprite.RenderPlain(self.pad_group)
        self.car_group = pygame.sprite.RenderPlain(self.car_group)
        self.trophy_group = pygame.sprite.RenderPlain(self.trophy_group)
        self.checkpoint_group = pygame.sprite.RenderPlain(self.checkpoint_group)
        # set up font
        self.font = pygame.font.Font(None, 40)
        self.screenmessage = self.font.render('', True, (255,0,0))
        # set up game window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))


    def render(self):
        self.screen.fill((0,0,0))# empty screen        
        self.pad_group.update(self.collisions)#change pad image if hit
        self.car_group.draw(self.screen)#draw car
        self.pad_group.draw(self.screen)
        self.trophy_group.draw(self.screen)
        self.checkpoint_group.draw(self.screen)
        self.screen.blit(self.font.render(self.screenmessage, True, (0,255,0)), (150, 700))
        pygame.display.flip()

    def reset(self):
        # reset the environment to initial state
        #return observation
        self.__init__()
        self.screenmessage = ''
        # give out current state
        state = (self.car.position[0], self.car.position[1], self.car.direction, self.car.speed)
        return state


    def calculate_reward(self):
        # calculate the reward based on the current state
        # reward is 0 if no collision
        # reward is -MAX_REWARD if collision
        # reward is MAX_REWARD if trophy is reached
        self.collisions = pygame.sprite.groupcollide(self.car_group, self.pad_group, False, False, collided = None)
        if self.collisions != {}:
            self.win_condition = False
            #timer_text = font.render("Crash!", True, (255,0,0))
            self.car.image = pygame.image.load(IMAGEPATH+'collision.png')
            #loss_text = win_font.render('Press Space to Retry', True, (255,0,0))
            self.car.MAX_SPEED = 0

        trophy_collision = pygame.sprite.groupcollide(self.car_group, self.trophy_group, False, True)
        if trophy_collision != {}:
            self.win_condition = True
            self.car.MAX_SPEED = 0
        
        # make sure center of car didnt leave the screen
        x_condition = self.car.rect.center[0] < 0 or self.car.rect.center[0] > WINDOW_WIDTH
        y_condition = self.car.rect.center[1] < 0 or self.car.rect.center[1] > WINDOW_HEIGHT
        if x_condition or y_condition:
            self.win_condition = False
            self.car.MAX_SPEED = 0
        # calculate dynamic reward
        self.reward_func()

        # return done
        return self.win_condition is not None

    def reward_func(self):
        reward = [0]
        # buffer around car
        buffered_car = scale_rect(self.car.rect, BUFFER_RATIO)
        ### if too close to to screen (left/right/bottom), buffer penalty if too close, 0 if not
        if (buffered_car.left < 0) or (buffered_car.right > WINDOW_WIDTH) or (buffered_car.bottom > WINDOW_HEIGHT): 
            reward.append(BUFFER_PENALTY*2)
        # if too close to pads as buffer penalty or 0
        close_miss_pad = [buffered_car.colliderect(pad.rect) for pad in self.pads]
        if np.any(close_miss_pad): reward.append(BUFFER_PENALTY)

        # ##distance to outside of closest pad (doenst work)
        #distance_pads = round(self.calculate_distance_to_pads(),1)
        
        # ### distance to trophy
        # distance_trophy = np.array(self.car.rect.center) - np.array(self.trophy.rect.center)
        # # normalize by screen width, height, i.e. between 0 and 1 * BUFFER_PENALTY
        # distance_trophy = distance_trophy * (0.2,0.8) #weighting x less than y
        # distance_trophy = round(1/(np.sum(distance_trophy / np.array([WINDOW_WIDTH, WINDOW_HEIGHT]))),1)
        # #reward.append(distance_trophy)

        ### if checkpoint n is reached, add constant to reward from there on
        if self.checkpoints[self.checkpoint_counter].rect.collidepoint(self.car.rect.center):
            self.checkpoint_reward = CHECKPOINT_REWARD*(self.checkpoint_counter+1)#needs to be 1-indexed
            if (self.checkpoint_counter < len(self.checkpoints)-1):#as long as not in final zone, next checkpoint has to be reached
                # checkpoints 1+2 are on same y, so add if one of them is reached
                    self.checkpoint_counter += 1
        reward.append(self.checkpoint_reward)

        ### euclidian distance to next checkpoint
        distance_checkpoint = np.array(self.car.rect.center) - np.array(self.checkpoints[self.checkpoint_counter].rect.center)
        # normalize by screen width, height, i.e. between 0 and 1 
        distance_checkpoint_print = (np.abs(distance_checkpoint) / np.array([WINDOW_WIDTH, WINDOW_HEIGHT]))
        # subtract from 1
        distance_checkpoint = 1-np.sum(distance_checkpoint_print)
        reward.append(distance_checkpoint)
        ### sum up all penalties
        self.reward = round(sum(reward),1)# + distance_trophy,1)# + distance_pads,1)

        # Overwrite if fail or success
        if self.win_condition is not None:
            self.reward = [-MAX_REWARD, MAX_PENALTY][int(self.win_condition)]
        self.screenmessage = f"{self.reward}, {distance_checkpoint_print}"

    def calculate_distance_to_pads(self):
        car_center = np.array(self.car.rect.center)
        distances = []
        for pad in self.pads:
            pad_rect = pad.rect
            pad_distances = [
                #np.linalg.norm(car_center - np.array([pad_rect.left, car_center[0]])),  # Distance to left edge
                #np.linalg.norm(car_center - np.array([pad_rect.right, car_center[0]])),  # Distance to right edge
                #np.linalg.norm(car_center - np.array([car_center[0], pad_rect.top])),  # Distance to top edge
                #np.linalg.norm(car_center - np.array([car_center[0], pad_rect.bottom]))  # Distance to bottom edge
                car_center[1]- pad_rect.top,
                car_center[1]- pad_rect.bottom
            ]
            distances.append(min(pad_distances))
            min_distance = min(distances)/100
        return min_distance
    
    def plot_reward_map(self):
        reward_map = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT))
        # plot the reward map
        for x in range(0, WINDOW_WIDTH):
            for y in range(0, WINDOW_HEIGHT):
                print(x)
                self.car.rect.center = (x,y)
                # skip if center inside any of the pad
                if not any([pad.rect.collidepoint(self.car.rect.center) for pad in self.pads]):                
                    self.reward_func()
                    reward_map[x,y] = self.reward
        return(reward_map)

    def step(self, action):
        # perform one step in the game logic
        # move car
        state = self.car.update(action)
        # check collision, calc reward
        done = self.calculate_reward()
        # print win/loss message
        if self.win_condition is not None:
            self.screenmessage = ['You fucked up', 'Finished!'][int(self.win_condition)]
            while self.message_printed == False:
                #print(self.screenmessage)
                self.message_printed = True
        return state, self.reward, done
    
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