import numpy as np
import pygame
import time
from math import radians, sin, cos, degrees

episode_count = 1000
learning_rate = 0.8
discount_factor = 0.99
bins = 36

def gen_reward(pos, vel, direction):
    #if(vel>7 or vel<-7): return -15     # too fast
    if(pos==bins//2 or pos==bins//2 -1): return 120     # at top
    if(pos==bins//2 +1 or pos==bins//2 -2): return 80    # close to top
    return bins//2 - 3*abs(bins//2-pos) # more positive closer to top

def pos_bin(angular_pos):
    return int(bins*angular_pos)//360
def vel_bin(angular_vel):
    return int(bins*angular_vel)//360


g = 9.8
a = 5 # force/mass
l = 5
dt = 0.1
demonstration = False

Q = np.zeros((bins, bins, 3))


for episode in range(episode_count+5):
    angular_pos = 0
    angular_vel = 0
    pos = pos_bin(angular_pos)
    vel = vel_bin(angular_vel)
    direction = 0

    if(episode == episode_count): 
        demonstration = True
        print("\n\nDEMONSTRATION\n\n")
        pygame.init()         # display screen
        screen = pygame.display.set_mode((500, 500))

    frame_count = 500
    for frame in range(frame_count):
        if demonstration: # show results
            time.sleep(0.05)
            screen.fill("black")
            rod_end_pos = [250+100*sin(radians(angular_pos)), 250+100*cos(radians(angular_pos))]

            pygame.draw.circle(screen, "grey", [250,250], 3)
            if(direction==2): pygame.draw.circle(screen, "red", rod_end_pos, 3)
            if(direction==0): pygame.draw.circle(screen, "green", rod_end_pos, 3)
            pygame.draw.line(screen, "white", [250,250], rod_end_pos )
            pygame.display.flip()

        if(np.random.uniform(0,1)<0.2 and not demonstration):
            direction = np.random.randint(0,3)  # explore
        else:
            direction = np.argmax(Q[pos, vel, :]) # greedy

        reward = gen_reward(pos, vel, direction)

        angular_pos = (angular_pos + angular_vel * dt + 360) % 360
        angular_acc = -3*g*sin(radians(angular_pos)) / l**2 + 3*(direction-1)*a / l
        angular_vel += degrees(angular_acc) * dt #+ np.random.uniform(-0.01,0.01)

        next_pos = pos_bin(angular_pos)
        next_vel = vel_bin(angular_vel)

        Q[pos, vel, direction] += learning_rate * (reward + discount_factor* np.max(Q[next_pos,next_vel,:] - Q[pos, vel, direction]))

        if(angular_vel > 180 or angular_vel < -180): break
        pos = next_pos
        vel = next_vel