# -*- coding: utf-8 -*-
"""
Late edited: March 2022

@author: shugh
"""

# fdtd_V1p0.py - Simple Exampel code for Q1 of PS5(Part A)

"""
1d FDTD Simulation for Ex nd Hy, simple dielectric, simple animation
Units for E -> E*sqrt(exp0/mu0), so E and H are comparible (and same for a plane wave)
No absorbing BC (ABC) - so will bounce off walls
Source (injected at a single space point) goes in both directions

Code is not vectorized, but is kept simple to show you what is going on

Set your graphics to Auto, e.g., within Spyder or
%matplotlib Auto
-- Can also add these two lines (works under windows)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')
"""

from pickle import FALSE
import numpy as np
import scipy.constants as constants
# comment these if any problems - sets graphics to auto
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'auto')

from matplotlib import cm, colors, rcParams
import matplotlib.pyplot as plt

rcParams['font.size'] = 18
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1


# Plotting config
time_pause = 0.8  # specifies time a frame is displayed in animation
save_plot = False  # 0 will not save graphics 
cycle = 100  # for graph updates

# initialize graph, fixed scaling for this first example
def init_plt1():
    plt.ylim((-0.7, 0.7))
    plt.xlim((0, n_xpts-1))    
    plt.axvline(x=n_xpts//2,color='r')  # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.pause(1)

def update_plot(i, cycle, im, Ex):
    if i % cycle != 0:
        return
    im.set_ydata(Ex[1:n_xpts-1])
    ax.set_title("frame time {}".format(i))
    plt.draw()
    plt.pause(time_pause) # sinsible pause to watch animation 

    if save_plot and i % 2*cycle == 0:
        plt.savefig("./figs/cycle_{}.pdf".format(i*100), dpi=800)
           

# Basic geometry and dielectric parameters
n_xpts = 801  # number of FDTD cells in x
n_steps = 2000  # number of FDTD tiem steps
c = constants.c  # speed of light in vacuum
fs = constants.femto  # 1.e-15 - useful for pulses 
tera = constants.tera  # 1.e12 - used for optical frequencues 
epsilon_0 = constants.epsilon_0
dx = 20e-9 #  FDTD grid size in space, in SI Units
dt = dx / (2 * c) # FDTD time step

Ex = np.zeros(n_xpts, dtype=np.float64)  # E array  
Hy = np.zeros(n_xpts, dtype=np.float64)  # H array

# Pulse parameters and points per wavelength"
isource = 200  # source position
spread = 2 * fs/dt  # 2 fs for this example
t0 = spread * 6
freq_in = 2*np.pi * 200 * tera  # incident (angular) frequency
w_scale = freq_in * dt
lam = 2*np.pi * c / freq_in  # near 1.5 microns
ppw = lam // dx  # points per wavelength
if ppw <= 15:
    raise Exception(f'Points per wavelength should be > 15 but got {ppw}')

# Pulse function
pulse_fn = lambda t: -np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*w_scale))

# TODO: obtain numerical reflection of ABC
def a_FDTD_loop_1D(nsteps, cycle):
    # Keep track of previous 2 values of E at boundary for absorbing BC
    Ex_1 = {'n-1' : 0,  'n-2' : 0}
    Ex_end = {'n-1' : 0,  'n-2' : 0}
    
    # loop over all time steps
    for i in range(nsteps+1):
        t = i-1  # iterative time dep pulse as source
        pulse = pulse_fn(t)

        # Update stored boundary points
        Ex_1['n-2'] = Ex_1['n-1']
        Ex_1['n-1'] = Ex[1]
        Ex_end['n-2'] = Ex_end['n-1']
        Ex_end['n-1'] = Ex[-2]

        # Update E
        Ex[1:-1] = Ex[1:-1] + 0.5 * (Hy[0:-2] - Hy[1:-1])
        Ex[isource] = Ex[isource] - 0.5 * pulse

        # Apply absorbing BCs
        Ex[0] = Ex_1['n-2']
        Ex[-1] = Ex_end['n-2']
                
        # Update H - note the offset
        Hy[0:-1] = Hy[0:-1] + 0.5 * (Ex[0:-1] - Ex[1:])

        # Update graph every cycle
        update_plot(i, cycle, im, Ex)

def b_FDTD_loop_1D(nsteps, cycle):
    # Keep track of previous 2 values of E at boundary for absorbing BC
    Ex_1 = {'n-1' : 0,  'n-2' : 0}
    Ex_end = {'n-1' : 0,  'n-2' : 0}
    
    # loop over all time steps
    for i in range(nsteps+1):
        t = i-1  # iterative time dep pulse as source
        Ex_source = pulse_fn(t)
        Hy_source = pulse_fn(t + 0.5)

        # Update stored boundary points
        Ex_1['n-2'] = Ex_1['n-1']
        Ex_1['n-1'] = Ex[1]
        Ex_end['n-2'] = Ex_end['n-1']
        Ex_end['n-1'] = Ex[-2]

        # Update E
        Ex[1:-1] = Ex[1:-1] + 0.5 * (Hy[0:-2] - Hy[1:-1])
        Ex[isource] = Ex[isource] - 0.5 * Hy_source

        # Apply absorbing BCs
        Ex[0] = Ex_1['n-2']
        Ex[-1] = Ex_end['n-2']
                
        # Update H - note the offset
        Hy[0:-1] = Hy[0:-1] + 0.5 * (Ex[0:-1] - Ex[1:])
        Hy[isource - 1] = Hy[isource - 1] - 0.5 * Ex_source

        # Update graph every cycle
        update_plot(i, cycle, im, Ex)
           

#%%
"Define first (only in this simple example) graph for updating Ex at varios times"
fig = plt.figure(figsize=(8,6))
ax = fig.add_axes([.18, .18, .7, .7])
[im] = ax.plot(np.arange(1, n_xpts-1), Ex[1:n_xpts-1], linewidth=2)
init_plt1()  # initialize, then we will just update the y data and title frame

"Main FDTD: time steps = nsteps, cycle for very simple animation"
# a_FDTD_loop_1D(n_steps, cycle)
b_FDTD_loop_1D(n_steps, cycle)
