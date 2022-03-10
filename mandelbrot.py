#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle
import math
import cmath
import numpy as np
import sys

resolution = 200
min_depth = 3
max_depth = 100
depth_step = 10

fig = plt.figure(figsize=(24, 24))

def iterate(z,c,curr_depth):
    global resolution, max_depth, should_plot, curr_points
    curr_depth -= 1
    z_new = z**2 + c
    c_mag = cmath.polar(c)[0]
    c_phase = cmath.polar(c)[1]
    magnitude = cmath.polar(z_new)[0]
    i = complex(0,1)

    in_cardoid = c_mag < cmath.polar(cmath.exp(i*c_phase/2) - cmath.exp(i*c_phase/4))[0]
    in_circle =  cmath.polar(c + complex(1,0))[0] < cmath.polar(0.25*(cmath.exp(i*c_phase)))[0]
    if in_cardoid or in_circle or (curr_depth <= 0 and magnitude <=2): 
        # np.append([c.real,c.imag],curr_points)
        curr_points.append([c.real,c.imag])
        return True
    
    elif (magnitude > 2):
        return False
    
    else:
        return iterate(z_new,c,curr_depth)

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    print(ix,iy)

    # assign global variable to access outside of function
    global zoom_coords
    zoom_coords.append((ix, iy))
    return

def onrelease(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata

    print(ix,iy)

    # assign global variable to access outside of function
    global zoom_coords
    zoom_coords.append((ix, iy))
    if len(zoom_coords) == 2:
        plt.ion()
        xmin = min(zoom_coords[0][0],zoom_coords[1][0])
        xmax = max(zoom_coords[0][0],zoom_coords[1][0])
        ymin = min(zoom_coords[0][1],zoom_coords[1][1])
        ymax = max(zoom_coords[0][1],zoom_coords[1][1])
        zoom_coords = []
        current_axis = plt.gca()
        current_axis.add_patch(Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='b', facecolor='none',zorder=100))
        plt.show()
        plt.pause(0.1)
        plot(xmin, xmax, ymin, ymax)
    else:
        zoom_coords = []
    return

cid_1 = fig.canvas.mpl_connect('button_press_event', onclick)
cid_2 = fig.canvas.mpl_connect('button_release_event', onrelease)

def plot(min_x, max_x, min_y, max_y):
    global max_depth, depth_step, curr_points   
    plt.clf()
    plt.ioff()
    points = [complex(x,y) for x in np.linspace(min_x,max_x,resolution,dtype='float128') for y in np.linspace(min_y,max_y,resolution,dtype='float128')]
    i = 0
    last_progress = 0

    for depth in range(min_depth,max_depth + 1,depth_step):
        # curr_points = np.array([])
        curr_points = []
        for point in points:
            progress = round(float(i)/(resolution*resolution*len(range(min_depth,max_depth + 1,depth_step))),3)
            if progress != last_progress:
                # sys.stdout.write("\r" + "\t" + str(progress*100) + "%")
                print(str(progress*100) + '%')
            last_progress = progress
            i+=1
            iterate(0,point,depth)
        curr_points = np.array(curr_points,dtype="float128")
        depth = float(depth)
        color = hsv_to_rgb([0.7*(1-depth/max_depth),1,1-depth/max_depth])
        # plt.scatter(curr_points[:,0],curr_points[:,1],s=int(25000/(resolution)),c=color,marker="s")
        plt.plot(curr_points[:,0],curr_points[:,1],markersize=(1300/(resolution)),color=color,marker="s",linestyle="None")
        # plt.plot(curr_points[:,0],curr_points[:,1],markersize=(1),color=color,marker=",",linestyle="None")
        ax = plt.gca()
        ax.set_facecolor('xkcd:blue')
        # plt.imshow(curr_points, interpolation='nearest')
        fig.savefig('mandelbrot_current.png')
        # im = plt.imread('mandelbrot_current.png')
        # plt.imshow(im, interpolation='nearest')

    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.axis('equal')
    plt.show()

def main():
    global zoom_coords
    zoom_coords = []
    min_x = -2
    max_x = 2
    min_y = -2
    max_y = 2
    # min_x = -0.7585
    # max_x = -0.7550
    # min_y = 0.0610
    # max_y = 0.0645
    plot(min_x, max_x, min_y, max_y)
        
if __name__ == "__main__":
    main()