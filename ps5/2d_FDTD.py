# -*- coding: utf-8 -*-
"""
ENPH 479 PS 5: Part 2 code: 2-dimensional FDTD

This file holds my code for a 2-dimensional Finite-Difference Time-Domain simulation to
solve Maxwell's equations. 

We experiment with looping, slicing, and Numba to find the optimal approach to solve the
problem.
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.constants as constants
import timeit

from numba import njit

# comment these if any problems - sets graphics to auto
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')

plt.rcParams['font.size'] = 15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1


# For animation updates - will slow down the loop to see Ex frames better
time_pause = 0.1
printgraph_flag = True  # print graph to pdf (1)
livegraph_flag = True  # update graphs on screen every cycle (1)
cycle = 100 # for graph updates

def graph(t, Ez, save_name=''):
    # main graph is E(z,y, time snapshops), and a small graph of E(t) as center
    plt.clf() # close each time for new update graph/colormap
    ax = fig.add_axes([.25, .25, .6, .6])   
    ax2 = fig.add_axes([.015, .8, .15, .15])   

    # 2d plot - several options, two examples below
    # img = ax.imshow(Ez)
    img = ax.contourf(Ez)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label('$Ez$ (arb. units)')

    # add labels to axes
    ax.set_xlabel('Grid Cells ($x$)')
    ax.set_ylabel('Grid Cells ($y$)')
     
    # dielectric box - comment if not using of course (if eps_box=1)
    if eps_box != 1:
        ax.vlines(X1,Y1,Y2,colors='r')
        ax.vlines(X2,Y1,Y2,colors='r')
        ax.hlines(Y1,X1,X2,colors='r')
        ax.hlines(Y2,X1,X2,colors='r')

    # add title with current simulation time step
    ax.set_title(f"Frame: {t}, $\epsilon_2 = ${eps_box}")

    # Small graph to see time development as a single point
    PulseNorm = np.asarray(PulseMonTime)*0.2
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1,'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title(r'$E_{in}(t)$')

    plt.draw()
    if save_name:
        plt.savefig(f"./figs/{save_name}_{t}.pdf", dpi=800)
    plt.pause(time_pause) # pause sensible value to watch what is happening

# Pulse parameters and points per wavelength
def check_ppw(n, freq_in, dx):
    # Function checks that there are enough points per wavelength for a given material
    v = c / n  # speed of EM wave in medium
    lam =  2*np.pi * v / freq_in  # wavelength
    ppw = lam // dx  # points per wavelength
    if ppw <= 15:
        raise Exception(f'Points per wavelength should be > 15 but got {ppw}')

# Pulse function
@njit(fastmath=True)
def pulse_fn(t):
    return np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))


# ---------------- Definintion of update functions ----------------

def DE_update_loop(Dz, Hy, Hx, Ez, ga):
    for x in range (1, n_xpts-1): 
        for y in range(1, n_ypts-1):
            Dz[x,y] =  Dz[x,y] + 0.5 * (Hy[x,y] - Hy[x-1,y] - Hx[x,y] + Hx[x,y-1]) 
            Ez[x,y] =  ga[x,y] * Dz[x,y]
    return Dz, Ez

def DE_update_slice(Dz, Hy, Hx, Ez, ga):
    Dz[1:-1, 1:-1] =  Dz[1:-1, 1:-1] + 0.5 * (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1] - Hx[1:-1, 1:-1] + Hx[1:-1, :-2]) 
    Ez[1:-1, 1:-1] =  ga[1:-1, 1:-1] * Dz[1:-1, 1:-1]
    return Dz, Ez

def H_update_loop(Hx, Hy, Ez):
    for x in range(n_xpts-1): 
        for y in range(n_ypts-1): 
            Hx[x,y] = Hx[x,y] + 0.5 * (Ez[x,y] - Ez[x,y+1])                       
            Hy[x,y] = Hy[x,y] + 0.5 * (Ez[x+1,y] - Ez[x,y])
    return Hx, Hy

def H_update_slice(Hx, Hy, Ez):
    Hx[:-1, :-1] = Hx[:-1, :-1] + 0.5 * (Ez[:-1,:-1] - Ez[:-1, 1:])                       
    Hy[:-1, :-1] = Hy[:-1, :-1] + 0.5 * (Ez[1:,:-1] - Ez[:-1, :-1])
    return Hx, Hy


# Main FDTD loop iterated over nsteps
def FDTD_loop(nsteps, cycle, ga):
    if use_slicing:
        DE_update_fn = DE_update_slice
        H_update_fn = H_update_slice
    else:
        DE_update_fn = DE_update_loop
        H_update_fn = H_update_loop

    if use_jit:
        DE_update_fn = njit()(DE_update_fn) 
        H_update_fn = njit()(H_update_fn) 

    Ez = np.zeros(grid_shape, dtype=np.float64)
    Dz = np.zeros(grid_shape, dtype=np.float64)
    Hx = np.zeros(grid_shape, dtype=np.float64)
    Hy = np.zeros(grid_shape, dtype=np.float64) 

    # loop over all time steps
    for t in range(nsteps):
        # iterate pulse (t is an integer, so dt steps)
        pulse = pulse_fn(t)

        # calculate Dz (Hy is diff sign to before with Dz term from curl eqs)
        Dz, Ez = DE_update_fn(Dz, Hy, Hx, Ez, ga)
    
        Dz[isource, jsource] = Dz[isource, jsource] + pulse  # soft source in simulation center
        Ez[isource, jsource] = ga[isource, jsource] * Dz[isource, jsource]

        # save one point in time just to see the transient
        EzMonTime1.append(Ez[isource, jsource]) 
        PulseMonTime.append(pulse) 

        # update H (could also do slicing - but let's make it clear just now)    
        Hx, Hy = H_update_fn(Hx, Hy, Ez)

        if livegraph_flag and t % cycle == 1: # simple animation
            graph(t, Ez, f'p3a_eps-{eps_box}')


# Booleans to determine how to solve the problem
use_slicing = False
use_jit = True

# Basic Geometry and Dielectric Parameters
n_xpts = n_ypts = 1000  # no of FDTD cells in x and y (dealing with a square region here)
nsteps = 1000 # total number of FDTD time steps

grid_shape = (n_xpts, n_ypts)
ga = np.ones(grid_shape, dtype=np.float64)
cb = np.zeros(grid_shape, dtype=np.float64)  # for spatially varying dielectric constant

EzMonTime1 = []
PulseMonTime = [] # two time-dependent field monitors

c = constants.c  # speed of light in vacuum
fs = constants.femto  # 1.e-15 - useful for pulses 
tera = constants.tera  # 1.e12 - used for optical frequencues

dx = 20.e-9 / 2  #  FDTD grid size in space, in SI Units
dt = dx/(2.*c)  # FDTD time step

isource = int(n_ypts/2)  # x position of pulse source
jsource = int(n_xpts/2)  # x position of pulse source
spread = 1 * fs/dt  # 2 fs for this example
t0 = 6 * spread
freq_in = 2*np.pi * 200*tera  # incident (angular) frequency
w_scale = freq_in * dt
eps_box = 9  # dielectric box (so 1 is just free space)
check_ppw(eps_box, freq_in, dx)

# simple fixed dielectric box coordinates
X1 = isource + 10
X2 = X1 + 40
Y1 = jsource + 10
Y2 = Y1 + 40
ga[X1:X2, Y1:Y2] = 1 / eps_box

# an array for x,y spatial points (with first and last points)
xs = np.arange(n_xpts)  
ys = np.arange(n_ypts)  

#%% Initial Run

printgraph_flag = True  # print graph to pdf (1)
livegraph_flag = True  # update graphs on screen every cycle (1)

# Booleans to determine how to solve the problem
use_slicing = False
use_jit = True

# set figure for graphics output
if livegraph_flag: 
    fig = plt.figure(figsize=(8,6))

# Main FDTD: time steps = nsteps, cycle for very simple animation
FDTD_loop(nsteps, cycle, ga)


#%% Timing Tests

printgraph_flag = False  # print graph to pdf (1)
livegraph_flag = False  # update graphs on screen every cycle (1)
nsteps = 1000

def run_time_test():
    print(f'nsteps: {nsteps}, npts: {n_xpts}')
    print(f'slicing: {use_slicing},  numba: {use_jit}')

    start = timeit.default_timer()      
    FDTD_loop(nsteps, cycle, ga)
    stop = timeit.default_timer()

    print(f"Time for FDTD simulation: {round(stop - start, 3)} s \n")


use_slicing = False
use_jit = False
nsteps = 100
print(f'multiply time by 10 for this case:')
run_time_test()

nsteps = 1000

use_slicing = True
use_jit = False
run_time_test()

use_slicing = False
use_jit = True
run_time_test()

use_slicing = True
use_jit = True
run_time_test()

