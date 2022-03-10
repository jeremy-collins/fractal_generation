#!/usr/bin/env python
import turtle as t
import time
import math

t.setup(2000,2000)
t.bgcolor("black")
t.pencolor("white")
t.speed(0)
t.setheading(-90)
t.pendown()

t.tracer(0, 0)

def left(dist):
    t.left(90)
    t.forward(dist)

def right(dist):
    t.right(90)
    t.forward(dist)

def invert(string):
    newStr = ''
    for char in string:
        if char == 'R':
             newStr += 'L'
        elif char == 'L':
            newStr += 'R'
        else:
            print("invalid string") 
    return newStr[::-1]

def genString(LRstring):
    global depth
    if len(LRstring) >= (2**(depth) - 1):
        return LRstring
    else:
        nextStr = LRstring + 'R' + invert(LRstring)
        return genString(nextStr)


for depth in range(1,24):
    print(depth)
    t.reset()
    t.setposition(200,400)
    t.setheading(-135 + 45*depth)
    t.pencolor("white")  
    t.pendown()
    t.colormode(1.0)
    dist = 2**(10 - 0.5*depth)  #if depth > 2 else 2**(10 - depth)
    # dist = 1
    folds = genString('R')
    # print(folds)
    for index,fold in enumerate(folds):
        if fold == 'R':
            right(dist)
        elif fold == 'L':
            left(dist) 
        color_speed = 2**(depth - 3)
        red_weight = 1
        green_weight = 1
        blue_weight = 1
        phase_offset = 0.3
        t.pencolor((red_weight*(0.5*math.sin(index/color_speed)+0.5), green_weight*(0.5*math.sin((index*(1 + phase_offset) + math.pi/3)/color_speed)+0.5), blue_weight*(0.5*math.sin((index*(1 - phase_offset) + math.pi*2/3)/color_speed)+0.5)))    
        # t.pencolor("white")  
    t.hideturtle()
    t.update()
    time.sleep(1)

# t.exitonclick()

# R
# R_R_L
# RRL_R_RLL
# RRLRRLL_R_RRLLRLL
# RRLRRLLRRRLLRLL_R_RRLRRLLLRRLLRLL