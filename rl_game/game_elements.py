import pygame
import math
import numpy as np
from rl_game.game_config import *


class CarSprite(pygame.sprite.Sprite):
    def __init__(self, image, position):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.image.load(image)
        self.rect = self.src_image.get_rect()
        self.rect.center = position
        self.position = np.array(position, dtype=float)  # Use a separate attribute for position
        self.speed = 0.2
        self.direction = 320
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.ACCELERATION = ACCELERATION
        self.TURN_ACCELERATION = TURN_ACCELERATION

    def update(self, action):
        # SIMULATION
        # action[0]: acceleration -1:back, 0:none, 1:forwards | action[1]: rotation, 1:left, -1:right
        # add acceleration to current speed
        self.speed += action[0] * self.ACCELERATION
        if self.speed > self.MAX_SPEED:
            self.speed = self.MAX_SPEED
        elif self.speed < self.MIN_SPEED:
            self.speed = self.MIN_SPEED
        # add change of direction to current direction
        self.direction += action[1] * self.TURN_ACCELERATION
        self.direction %= 360  # needs remapping to [0,359] because can take any value
        # calculate new position
        rad = self.direction * math.pi / 180
        self.position[0] += -self.speed * math.sin(rad)#x
        self.position[1] += -self.speed * math.cos(rad)#y
        # update rect center
        self.rect.center = self.position.astype(int)
        # rotate image + rect accordingly
        self.image = pygame.transform.rotate(self.src_image, self.direction)
        self.rect = self.image.get_rect(center=self.rect.center)

class PadSprite(pygame.sprite.Sprite):
    def __init__(self, position, width, height=25):
        super(PadSprite, self).__init__()
        self.image = pygame.Surface((width, height))
        self.image.fill((128, 128, 128))  # Fill the pad with a color (red in this case)
        self.rect = self.image.get_rect()
        self.rect.center = position
class CheckpointSprite(pygame.sprite.Sprite):
    def __init__(self, position, width=150, height=25):
        super(CheckpointSprite, self).__init__()
        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.image.fill((255, 204, 203, 255))  # Fill with white color and set transparency

        # Create a dotted pattern
        dot_spacing = 5
        for x in range(0, width, dot_spacing):
            for y in range(0, height, dot_spacing):
                self.image.set_at((x, y), (0, 0, 0, 0))  # Set dots to be fully transparent
        self.rect = self.image.get_rect()
        self.rect.center = position

class Trophy(pygame.sprite.Sprite):
    def __init__(self, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(IMAGEPATH+'trophy.png')
        self.rect = self.image.get_rect()
        self.rect.center = position
    def draw(self, screen):
        screen.blit(self.image, self.rect)
