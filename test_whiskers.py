import pygame
from pygame.locals import *
from rl_game.racegame import CarSprite, PadSprite


pad = PadSprite(position=(250,150),width=50)
pad_g = pygame.sprite.RenderPlain(pygame.sprite.Group(pad))

car = CarSprite(image='Race_Game/images/car.png',position=(250, 220))
car_g = pygame.sprite.RenderPlain(pygame.sprite.Group(car))

VIEWFACTOR=5
view = car.rect.width*VIEWFACTOR
view_c = 0.75*view

def compute_whisker_hits(car, pad):
    whisker_coords = [(car.position, (car.position[0], car.position[1]-view)),#top
                    (car.position, (car.position[0]+view, car.position[1])),#right
                    (car.position, (car.position[0], car.position[1]+view)),#bottom
                    (car.position, (car.position[0]-view, car.position[1])),#left  
                    (car.position, (car.position[0]-view_c, car.position[1]-view_c)),#top left
                    (car.position, (car.position[0]+view_c, car.position[1]-view_c)),#top right
                    (car.position, (car.position[0]-view_c, car.position[1]+view_c)),#bottom left
                    (car.position, (car.position[0]+view_c, car.position[1]+view_c))] #bottom right

    clipped_whiskers = []
    for w in whisker_coords:
        clipped_whisker = pad.rect.clipline(*w)
        clipped_whiskers.append(clipped_whisker)
            
    return whisker_coords, clipped_whiskers



width = 1000
height = 500
line_color = (255, 0, 0)

screen=pygame.display.set_mode((width,height))
clock = pygame.time.Clock()
while True:
    clock.tick(2)
    screen.fill((0,0,0))# empty screen
    car_g.update([0,0])
    whisker_coords, clipped_whiskers = compute_whisker_hits(car, pad)
    car_g.draw(screen)
    pad_g.draw(screen)
    for w in whisker_coords:
        pygame.draw.line(screen, "white", *w)
    for wc in clipped_whiskers:
        if wc != tuple():#drop empty whiskers (i.e. no object in reach)
            pygame.draw.line(screen, "red", *wc)
            pygame.draw.circle(screen, "red", wc[0],4,4)
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_ESCAPE]==1:
            pygame.quit()

pygame.quit()