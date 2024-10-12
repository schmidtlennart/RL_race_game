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
    TURN_ACCELERATION = 14

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
        # self.observation_space = <gym.space>
        # self.action_space = <gym.space>

        # Initialize variables
        self.win_condition = None 
        self.t0 = time.time()

        # create obstacles
        pads_list = [(0,10), (600,10), (1100,10), (100,150), (600,150), (100,300), (800,300), (400,450), (700,450), (200,600), (900,600), (400,750), (800,750)]
        self.pads = [PadSprite(pad) for pad in pads_list]
        # create trophy
        self.trophy = Trophy((920, 720))
        # create car
        self.car = CarSprite(IMAGEPATH+'car.png', (10, 730))
        # render obstacles, car, trophy
        self.pad_group = pygame.sprite.RenderPlain(*self.pads)
        self.car_group = pygame.sprite.RenderPlain(self.car)
        self.trophy_group = pygame.sprite.RenderPlain(self.trophy)
        

    def init_render(self):
        pygame.init()
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
        # screen.blit(win_text, (250, 700))
        # screen.blit(loss_text, (250, 700))
        pygame.display.flip()

    def reset(self):
        # reset the environment to initial state
        #return observation    
        pass

    def check_collision(self):
        self.collisions = pygame.sprite.groupcollide(self.car_group, self.pad_group, False, False, collided = None)
        if self.collisions != {}:
            self.win_condition = False
            #timer_text = font.render("Crash!", True, (255,0,0))
            self.car.image = pygame.image.load(IMAGEPATH+'collision.png')
            loss_text = win_font.render('Press Space to Retry', True, (255,0,0))
            self.car.MAX_SPEED = 0

        trophy_collision = pygame.sprite.groupcollide(self.car_group, self.trophy_group, False, True)
        if trophy_collision != {}:
            self.win_condition = True
            #timer_text = font.render("Finished!", True, (0,255,0))
            self.car.MAX_SPEED = 0

    def step(self, action):
        # perform one step in the game logic
        # move car
        observation, reward, done, info = self.car.update(action)
        # check collision
        self.check_collision()
        return observation, reward, done, info   
    


def pressed_to_action(keytouple):
    action_turn = 0.
    action_acc = 0.
    key = None
    if keytouple[pygame.K_DOWN] == 1:  # back
        key = "back"
        action_acc -= 1
    if keytouple[pygame.K_UP] == 1:  # forward
        key = "forward"
        action_acc += 1
    if keytouple[pygame.K_LEFT] == 1:  # left
        key = "left"
        action_turn += 1
    if keytouple[pygame.K_RIGHT] == 1:  # right
        key = "right"
        action_turn -= 1
    # ─── KEY IDS ─────────
    # arrow forward   : 273
    # arrow backwards : 274
    # arrow left      : 276
    # arrow right     : 275
    print(f"KEY PRESSED: " + str(key))
    print(action_acc, action_turn)
    return np.array([action_acc, action_turn])


environment = RaceEnv()
environment.init_render()
run = True

while run:
    # set game speed to 30 fps
    environment.clock.tick(30)
    # ─── CONTROLS ───────────────────────────────────────────────────────────────────
    # end while-loop when window is closed or escape key is pressed
    get_event = pygame.event.get()
    for event in get_event:
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            run = False  
    # get pressed keys, generate action
    get_pressed = pygame.key.get_pressed()
    action = pressed_to_action(get_pressed)
    # calculate one step
    environment.step(action)
    # render current state
    environment.render()
pygame.quit()

    # font = pygame.font.Font(None, 75)
    # win_font = pygame.font.Font(None, 50)
    # win_text = font.render('', True, (0, 255, 0))
    # loss_text = font.render('', True, (255, 0, 0))