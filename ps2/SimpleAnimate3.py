# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 13:30:50 2022

@author: shugh

Some simple animation examples, mainly from the web 
"""

#%% Example 1
import matplotlib.pyplot as plt
import time
import random
 
ysample = random.sample(range(-50, 50), 100)
 
xdata = []
ydata = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line, = axes.plot(xdata, ydata, 'r-')
 
Q1=plt.quiver(0,0,ysample[0],-1,width=0.008,color='b',  
          headwidth=8.,headlength=3.,headaxislength=3,scale=40)

for i in range(100):
    Q1.remove()
    xdata.append(i)
    ydata.append(ysample[i])
    line.set_xdata(xdata)
    line.set_ydata(ydata)

    Q1=plt.quiver(i,ysample[i],1,-1,width=0.008,color='b',  
             headwidth=8.,headlength=4.,headaxislength=3,scale=25)
    # no direction on v, so just showing a simple example 

    plt.draw()
    plt.pause(1e-17)
    time.sleep(0.2) # similar to plt.pause here - just testing
    


#%% Example 2

"""
This simpel animation is taken from this sourse, and you can find plenty of other example online
# Source:https://learndataanalysis.org/a-basic-example-how-to-create-animation-matplotlib-tutorial/
# https://www.youtube.com/watch?v=dOKHY_PUvqU&ab_channel=JieJenn
"""

import numpy as np
from matplotlib.animation import FuncAnimation

x_data = []
y_data = []

fig, ax = plt.subplots()
ax.set_xlim(0, 105)
ax.set_ylim(0, 12)
line, = ax.plot(0, 0)

def animation_frame(i):
	x_data.append(i * 10)
	y_data.append(i)

	line.set_xdata(x_data)
	line.set_ydata(y_data)
	return line, 

# see https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 10, 0.2), interval=20)
plt.show()

#%% Example 3

"""
This example also uses matplotlib.animation
Source: https://riptutorial.com/matplotlib/example/23558/basic-animation-with-funcanimation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

TWOPI = 2*np.pi

fig, ax = plt.subplots()

t = np.arange(0.0, TWOPI, 0.001)
s = np.sin(t)
l = plt.plot(t, s)

ax = plt.axis([0,TWOPI,-1,1])

redDot, = plt.plot([0], [np.sin(0)], 'ro')

def animate(i):
    redDot.set_data(i, np.sin(i))
    return redDot,

# create animation using the animate() function
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(0.0, TWOPI, 0.1), \
                                      interval=10, blit=True, repeat=True)

plt.show()

#%% Example 4

"""
Simple example that moves the y data up and down
Source: https://stackoverflow.com/questions/37111571/how-to-pass-arguments-to-animation-funcanimation
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax1 = plt.subplots(1,1)

def animate(i,argu):
 #   print(i, argu)

    #graph_data = open('example.txt','r').read()
    graph_data = "1, 1 \n 2, 4 \n 3, 9 \n 4, 16 \n"
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y)+np.sin(2.*np.pi*i/10))
        ax1.clear()
        ax1.plot(xs, ys)
        plt.grid()

ani = animation.FuncAnimation(fig, animate, fargs=[5],interval = 10)
plt.show()

#%% Example 5

"""
Updated set of points and an animated gif save at end
Source: https://linuxtut.com/en/3c66781f41884694838b/
"""


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


fig = plt.figure()
ax = fig.add_subplot(111)

def update(frame):
    ax.plot(frame, 0, "o")

anim = FuncAnimation(fig, update, frames=range(8), interval=200)

# this will creat an animated gif - may complain but still  seems to work on windows
#anim.save("c01.gif", writer="imagemagick")  - so try this
#writergif = animation.PillowWriter(fps=40) 
#anim.save("test.gif", writer=writergif)
plt.show()
