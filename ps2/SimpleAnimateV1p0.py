# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25  2021

@author: shugh

This is a very simple example of VERY basic animation with matplotlib

There are far better ways to do this in Python, so feel free to experiment, but
it is better to learn a few simple things well (unless you are already experienced
with how to make simple animation). We also want to keep the graphics compatible for
creating nice graphs with TeX, which is more important for us.


Last Updated: Jan 19, 2022

# suggest adding this to consule or change your Spyder setting to "auto" on graphs
%matplotlib auto

"""

import numpy as np
import matplotlib.pyplot as plt

#import matplotlib.animation as animation #not using here

# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 20})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return 

# for simple graphics/animation of analytic SHO solution
def go():
    cycle=2; ic=0; h=0.05
    niceFigure(False) #can delete false if you have TeX installed
#    fig = plt.figure(figsize=(20,10))
    fig = plt.figure(figsize=(10,5))
#    plt.ion()
    ax = fig.add_axes([.18, .18, .33, .66])
    ax.set_ylim(-1.2,1.2)
    ax.set_xlim(-1.2,1.2)
    ax.set_xlabel('$x$ (arb. units)')     # add labels
    ax.set_ylabel('$y$ (arb. units)')
    # ax2 = fig.add_axes([.18, .18, .33, .66])
    # ax2.set_ylim(-1.2,1.2)
    # ax2.set_xlim(-1.2,1.2)
    x, v = 1.0, 0.0         # initial values
    line, = ax.plot( x, v,'xr', markersize=13) # Fetch the line object
    # line2, = ax2.plot( x, v,'xr', markersize=13) # Fetch the line object
    T0 = 2.*np.pi
    t=0.
    tpause = 0.1 # delay within animation 
                 #(though the minimum depends on your specs)
    plt.pause(2) # pause for 2 seconds before start

    while t<T0*5:        # loop for 5 periods
        
        v = -np.sin(t)
        # so your leapfrog or RK4 call could be here for example ...
        #xl, vl = leapfrog(oscillator, xl, vl, t, h) # solve this baby
 
        # you probably want to downsample if dt is too small (every cycle)
        if (ic % cycle == 0): # very simple animate (update data with pause)
            ax.set_title("frame time {}".format(ic)) # show current time on graph
            line.set_xdata(x)
            line.set_ydata(v)

            # plt.sca(ax2)
            # ax2.set_title("frame time {}".format(ic)) # show current time on graph
            # line2.set_xdata(x)
            # line2.set_ydata(v)

            plt.draw() # may not be needed (depends on your set up)
            plt.pause(tpause) # pause to see animation as code v. fast
           
        t  = t + h # loop time
        ic = ic + 1 # simple integer counter that migth be useful 

go()
