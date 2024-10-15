#initialize the screen
import pygame, math, time
import numpy as np
from pygame.locals import *
import gym
#reset
# game iteration
# playstep
# reward

### ToDO
# remove car+trophy groups
# 1. Implement the reset function

WINDOW_WIDTH, WINDOW_HEIGHT = 1024, 768
IMAGEPATH = 'Race_Game/images/'

class CarSprite(pygame.sprite.Sprite):
    MAX_SPEED = 8
    ACCELERATION = 1
    TURN_ACCELERATION = 5

    def __init__(self, image, position):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load(image)
        self.position = position
        self.speed = self.direction = 0
        self.k_left = self.k_right = self.k_down = self.k_up = 0
    
    def update(self, action):
        #SIMULATION
        # action[0]: acceleration | action[1]: rotation
        # add acceleration to current speed
        self.speed += action[0]*self.ACCELERATION
        if abs(self.speed) > self.MAX_SPEED:
            self.speed = self.MAX_SPEED if self.speed > 0 else -self.MAX_SPEED
        # add change of direction to current direction
        self.direction += action[1]*self.TURN_ACCELERATION
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
        observation, reward, done, info = (self.position, self.direction) , 0., False, {}
        return observation, reward, done, info

class PadSprite(pygame.sprite.Sprite):
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
    def __init__(self,env_config={}):
        # Initialize variables
        self.win_condition = None 
        self.t0 = time.time()

        # create obstacles
        pads_list = [(0,10), (600,10), (1100,10), (100,150), (600,150), (100,300), (800,300), (400,450), (700,450), (200,600), (900,600), (400,750), (800,750)]
        self.pads = [PadSprite(pad) for pad in pads_list]
        # create trophy
        self.trophy = Trophy((285,0))
        # create car
        self.car = CarSprite(IMAGEPATH+'car.png', (10, 730))
        # render obstacles, car, trophy
        self.pad_group = pygame.sprite.RenderPlain(*self.pads)
        self.car_group = pygame.sprite.RenderPlain(self.car)
        self.trophy_group = pygame.sprite.RenderPlain(self.trophy)
        

    def init_render(self):
        pygame.init()
        # set up font
        self.font = pygame.font.Font(None, 75)
        self.screenmessage = self.font.render('', True, (255,0,0))
        # set up game window
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        # set fps timer
        self.clock = pygame.time.Clock()


    def render(self):
        self.screen.fill((0,0,0))        
        self.pad_group.update(self.collisions)
        self.car_group.draw(self.screen)
        self.pad_group.draw(self.screen)
        self.trophy_group.draw(self.screen)
        self.screen.blit(self.screenmessage, (250, 700))
        pygame.display.flip()

    def reset(self):
        # reset the environment to initial state
        #return observation
        self.__init__()
        self.screenmessage = self.font.render('', True, (255,0,0))


    def check_collision(self):
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
            #timer_text = font.render("Finished!", True, (0,255,0))
            self.car.MAX_SPEED = 0
        
        # make sure car didnt leave the screen
        if not self.screen.get_rect().colliderect(self.car.rect):
            self.win_condition = False
            self.car.MAX_SPEED = 0

    def step(self, action):
        # perform one step in the game logic
        # move car
        observation, reward, done, info = self.car.update(action)
        # check collision
        self.check_collision()
        # print win/loss message
        if self.win_condition is not None:
            self.screenmessage = self.font.render(['You fucked up', 'Finished!'][int(self.win_condition)], True, (0,255,0))
        return observation, reward, done, info   
    


def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.
    if keytouple[pygame.K_DOWN] == 1:  # back
        action_acc -= 1
    if keytouple[pygame.K_UP] == 1:  # forward
        action_acc += 1
    if keytouple[pygame.K_LEFT] == 1:  # left
        action_turn += 1
    if keytouple[pygame.K_RIGHT] == 1:  # right
        action_turn -= 1
    # ─── KEY IDS ─────────
    # arrow forward   : 273
    # arrow backwards : 274
    # arrow left      : 276
    # arrow right     : 275
    return np.array([action_acc, action_turn])


environment = RaceEnv()
environment.init_render()
run = True

while run:
    # set game speed to 30 fps
    environment.clock.tick(30)
    # ─── CONTROLS ───────────────────────────────────────────────────────────────────
    get_event = pygame.event.get()
    # get pressed keys, generate action
    pressed_keys = pygame.key.get_pressed()
    action = pressed_to_action(pressed_keys)
    # end while-loop when window is closed or escape key is pressed
    for event in get_event:
        if event.type == pygame.QUIT or pressed_keys[pygame.K_ESCAPE]==1:
            run = False
        # reset the game when Failed or Won and space bar is pressed
        if (environment.win_condition is not None) and pressed_keys[pygame.K_SPACE] == 1:
            environment.reset()

    # calculate one step
    observation, reward, done, info = environment.step(action)
    # render current state
    environment.render()
pygame.quit()
