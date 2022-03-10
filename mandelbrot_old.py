#!/usr/bin/env python
import matplotlib.pyplot as plt
import math
import cmath
import numpy as np


resolution = 100
min_depth = 49
max_depth = 50
depth_step = 5
# min_x = -0.7388
# max_x = -0.7365
# min_y = 0.1737
# max_y = 0.1762
min_x = -3
max_x = 2
min_y = -2
max_y = 2

plt.figure(figsize=(24, 24))
def iterate(z,c,curr_depth):
    global resolution,max_depth,depth
    curr_depth -= 1
    z_new = z**2 + c
    c_mag = cmath.polar(c)[0]
    c_phase = cmath.polar(c)[1]
    magnitude = cmath.polar(z_new)[0]
    i = complex(0,1)
    # print(z_new)

    in_cardoid = c_mag < cmath.polar(cmath.exp(i*c_phase/2) - cmath.exp(i*c_phase/4))[0]
    in_circle =  cmath.polar(c + complex(1,0))[0] < cmath.polar(0.25*(cmath.exp(i*c_phase)))[0]
    if in_cardoid or in_circle or curr_depth <= 0 and magnitude < 10:
        red_weight = 1
        green_weight = 1
        blue_weight = 1
        color_speed = 1
        ### TODO: add points to matrix and graph all at once
        
        plt.scatter(c.real,c.imag,s=int(25000/(resolution)),c=[[red_weight*(0.5*math.sin(depth/max_depth/color_speed)+0.5), green_weight*(0.5*math.sin((depth/max_depth + math.pi/3)/color_speed)+0.5), blue_weight*(0.5*math.sin((depth/max_depth + math.pi*2/3)/color_speed)+0.5)]],marker="s")
        return True
    
    elif (curr_depth <= 0 and magnitude > 10) or (magnitude > 1000):
        return False
    
    else:
        return iterate(z_new,c,curr_depth)


def main():
    global max_depth, depth
    
    points = [complex(x,y) for x in np.linspace(min_x,max_x,resolution) for y in np.linspace(min_y,max_y,resolution)]
    i=0
    last_progress = 0

    for depth in range(min_depth,max_depth + 1,depth_step):
        for point in points:
            progress = round(float(i)/(resolution*resolution*(max_depth - min_depth)*depth_step),3)
            if progress != last_progress:
                print(str(progress*100) + '%')
            last_progress = progress
            i+=1
            iterate(0,point,depth)

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()