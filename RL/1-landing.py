"""
states based on discretized:
Height
velocity


make reward func

"""

import time
import numpy as np
import pygame


# reward from 
def gen_reward(hbin, vbin, power):

    vbin -= bin_count//2
    if(hbin==0 and vbin <-2): 
        print("crash")
        return -50
    if(hbin==0 and vbin == -2): 
        print("success")
        return 50 
    if(hbin==0 and vbin >= -1): 
        print("smooth landing")
        return 100
    if(hbin >= 2*bin_count//3 and power==1): 
        print(" high")
        return -10 
    if(hbin > bin_count-1): 
        print("too high")
        return -100 
    
    if(vbin < -2): 
        print("too fast")
        return -10 
    if(vbin == -2): 
        print("slightly fast")
        return -5
    if(vbin == -1): 
        print("good speed")
        return 10
    if(vbin > 0 and power==1): 
        print("going wrong way")
        return -20
    return 0

# continuous height to disrete bins
hbound = 300
def height_bin(height):
    if(height<=0): hbin=0
    else: hbin = (height*bin_count)//hbound + 1

    if(hbin>bin_count-1): hbin = bin_count-1
    return int(hbin)

# continuous velocity to discrete bins
speed_bound = 30
def velocity_bin(velocity):
    vbin = (velocity+speed_bound)*bin_count//(2*speed_bound) 
    if(vbin<0): vbin=0
    if(vbin>bin_count-1): vbin= bin_count-1
    return int(vbin)

a = 30
g = -9.8 # m/s^2
velocity = 0 # m/s
height = 200 # m
power = 0 # 0 or 1
dt = 0.2

bin_count = 10 
episode_count = 1000
learning_rate = 0.8
discount_factor = 0.98

Q = np.zeros((bin_count, bin_count, 2))

demonstration = False

for episode in range(episode_count+5):
    if(episode == episode_count): 
        demonstration = True
        print("\n\nDEMONSTRATION\n\n")
        # display screen
        pygame.init()
        screen = pygame.display.set_mode((500, 500))

    height = np.random.randint(150,200)
    velocity = np.random.randint(-10,10)
    hbin = height_bin(height)
    vbin = velocity_bin(velocity)

    #print("\n-----New Episode------- with height ", height, " and vel ", velocity)


    while(True):
        if demonstration: # show results
            time.sleep(0.2)
            screen.fill("black")
            pygame.draw.circle(screen, "white", np.array([250,450-height]), 15)
            if(power==1): pygame.draw.circle(screen, "yellow", np.array([250,450-height+10]), 10)
            pygame.draw.line(screen, "gray", [0,450], [500,450])
            pygame.display.flip()

        # chose action: 
        if(np.random.uniform(0,1) < 0.2 and not demonstration):
            power = np.random.randint(0,2)  #explore randomly
        else:
            power = np.argmax(Q[hbin, vbin, :]) # greedy choice

        reward = gen_reward(hbin, vbin, power) #reward and next step
        height += velocity
        velocity += (g+ a*power) * dt 
        next_hbin = height_bin(height)
        next_vbin = velocity_bin(velocity)
        #print("[ ", hbin, vbin, " ]")
        #print("height: ", round(height,2), "\tvel: ", round(velocity,2), "\tpower ", power, "\treward ", reward)

        # update Q values
        Q[hbin, vbin, power] = Q[hbin, vbin, power] + learning_rate * (discount_factor*np.max(Q[hbin,vbin,:]) + reward -Q[hbin, vbin, power])

        if(hbin==0 or height> 1.5*hbound): break
  
        hbin = next_hbin
        vbin = next_vbin

