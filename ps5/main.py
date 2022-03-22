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

rcParams['font.size'] = 10
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1


# Plotting config
time_pause = 0.8  # specifies time a frame is displayed in animation
save_plot = False  # 0 will not save graphics 
cycle = 100  # for graph updates
animate_flag = True  # specify whether or not to animate result

# initialize graph, fixed scaling for this first example
def init_plt_1():
    plt.ylim((-0.7, 0.7))
    plt.xlim((0, n_xpts-1))    
    plt.axvline(x=n_xpts//2,color='r')  # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.pause(1)

def init_plt_1c(dielectric_start, dielectric_end):
    plt.ylim((-0.7, 0.7))
    plt.xlim((0, n600))    
    plt.axvline(x=dielectric_start,color='r')  # Vert line separator
    plt.axvline(x=dielectric_end,color='r')  # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.pause(1)

def update_plot(i, cycle, Ex):
    if i % cycle != 0:
        return
    im.set_ydata(Ex[1:n_xpts-1])
    ax.set_title("frame time {}".format(i))
    plt.draw()
    plt.pause(time_pause) # sinsible pause to watch animation 

    if save_plot and i % 2*cycle == 0:
        plt.savefig("./figs/cycle_{}.pdf".format(i*100), dpi=800)

def plot_1c(E_in, E_t, E_r):
    # Plotting function for part 1 (c)
    fig = plt.figure(figsize=(8,6))

    # Plot electric fields over time
    ax1 = fig.add_axes([.2, .6, .6, .3])
    ax1.plot(range(n_steps + 1), E_in, 'b-', label=r'$E_{in}$')
    ax1.plot(range(n_steps + 1), E_t, 'r-', label=r'$E_{t}$')
    ax1.plot(range(n_steps + 1), E_r, 'g-', label=r'$E_{r}$')
    ax1.set_xlim(0, 3000)
    ax1.set_ylabel(r'$E_x$')
    ax1.set_xlabel(r'Time $(\Delta t)$')
    ax1.legend(loc='upper right')


    # Plot reflecred and transmitted in frequency domain
    ax2 = fig.add_axes([.2,.2,.6,.3])
    E_in_f = np.fft.rfft(E_in, norm='ortho')
    E_t_f = np.fft.rfft(E_t, norm='ortho')
    E_r_f = np.fft.rfft(E_r, norm='ortho')
    freq = np.fft.rfftfreq(E_in.size, d=dt) / tera

    T = np.abs(E_t_f/ E_in_f)**2
    R = np.abs(E_r_f)**2 / np.abs(E_in_f)**2
    ax2.plot(freq, T, 'r', label='T')
    ax2.plot(freq, R, 'b', label='R')
    ax2.plot(freq, T+R,'g' , label='Sum')
    ax2.set_xlim(100, 300)
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel(r'$T, R$')
    ax2.set_xlabel(r'$\omega /2\pi$ (THz)')
    ax2.legend(loc='lower right')
    
    plt.show()

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
    if animate_flag:
        init_plt_1()

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
        if animate_flag:
            update_plot(i, cycle, Ex)

def b_FDTD_loop_1D(nsteps, cycle):
    if animate_flag:
        init_plt_1()

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
        if animate_flag:
            update_plot(i, cycle, Ex)

def c_FDTD_loop_1D(nsteps, cycle):
    epsilon = 9
    L = 1e-6  # length of dielectric
    thickness_idx = int(L / dx)  # thickness in terms of x indices
    dielectric_start = 300  # initial position of dielectric
    permitivity_coeff = np.ones(n_xpts)
    permitivity_coeff[dielectric_start : dielectric_start + thickness_idx] = 1 / epsilon

    if animate_flag:
        init_plt_1c(dielectric_start, dielectric_start+thickness_idx)

    E_in = np.empty(n_steps+1)  # incident field over time
    E_t = np.empty(n_steps+1)  # transmitted field over time
    E_r = np.empty(n_steps+1)  # refelcted field over time

    # Keep track of previous 2 values of E at boundary for absorbing BC
    Ex_1 = {'n-1' : 0,  'n-2' : 0}
    Ex_end = {'n-1' : 0,  'n-2' : 0}

    # loop over all time steps
    for i in range(nsteps+1):
        t = i-1  # iterative time dep pulse as source
        Ex_source = pulse_fn(t)
        Hy_source = pulse_fn(t + 0.5)

        E_in[i] = pulse_fn(t)
        E_t[i] = Ex[dielectric_start + thickness_idx + 50]
        E_r[i] = Ex[isource - 50]

        # Update stored boundary points
        Ex_1['n-2'] = Ex_1['n-1']
        Ex_1['n-1'] = Ex[1]
        Ex_end['n-2'] = Ex_end['n-1']
        Ex_end['n-1'] = Ex[-2]

        # Update E
        Ex[1:-1] = Ex[1:-1] + 0.5 * permitivity_coeff[1:-1] * (Hy[0:-2] - Hy[1:-1])
        Ex[isource] = Ex[isource] - 0.5 * Hy_source

        # Apply absorbing BCs
        Ex[0] = Ex_1['n-2']
        Ex[-1] = Ex_end['n-2']

        # Update H - note the offset
        Hy[0:-1] = Hy[0:-1] + 0.5 * (Ex[0:-1] - Ex[1:])
        Hy[isource - 1] = Hy[isource - 1] - 0.5 * Ex_source

        # Update graph every cycle
        if animate_flag:
            update_plot(i, cycle, Ex)
    
    return E_in, E_t, E_r

#%%

#  Define first (only in this simple example) graph for updating Ex at varios times
if animate_flag:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), Ex[1:n_xpts-1], linewidth=2)

"Main FDTD: time steps = nsteps, cycle for very simple animation"
# # initialize, then we will just update the y data and title frame
# a_FDTD_loop_1D(n_steps, cycle)
# b_FDTD_loop_1D(n_steps, cycle)

run = False
n_steps = 30000
if run:
    # cycle = n_steps
    E_in, E_t, E_r = c_FDTD_loop_1D(n_steps, cycle)
    np.save('E_in', E_in)
    np.save('E_t', E_t)
    np.save('E_r', E_r)
else:
    E_in = np.load('E_in.npy')
    E_t = np.load('E_t.npy')
    E_r = np.load('E_r.npy')

plot_1c(E_in, E_t, E_r)

