# -*- coding: utf-8 -*-
"""





"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as constants
from mpi4py import MPI
from numba import njit

plt.rcParams['font.size'] = 15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    PulseNorm = PulseMonTime[:t+1] * 0.2
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(EzMonTime1[:t+1],'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title(r'$E_{in}(t)$')

    plt.draw()
    plt.savefig(f"./figs/{save_name}_{t}.pdf", dpi=800)

# Pulse function
@njit(fastmath=True)
def pulse_fn(t):
    return np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))

def DE_update_loop(Dz, Hy, Hx, Ez, ga):
    for x in range(1, nx): 
        for y in range(1, n_ypts-1):
            Dz[x,y] =  Dz[x,y] + 0.5 * (Hy[x,y] - Hy[x-1,y] - Hx[x,y] + Hx[x,y-1]) 
            Ez[x,y] =  ga[x,y] * Dz[x,y]
    return Dz, Ez

def DE_update_slice(Dz, Hy, Hx, Ez, ga):
    Dz[1:-1, 1:-1] =  Dz[1:-1, 1:-1] + 0.5 * (Hy[1:-1, 1:-1] - Hy[:-2, 1:-1] - Hx[1:-1, 1:-1] + Hx[1:-1, :-2]) 
    Ez[1:-1, 1:-1] =  ga[1:-1, 1:-1] * Dz[1:-1, 1:-1]
    return Dz, Ez

def H_update_loop(Hx, Hy, Ez):
    for x in range(nx-1): 
        for y in range(n_ypts-1): 
            Hx[x,y] = Hx[x,y] + 0.5 * (Ez[x,y] - Ez[x,y+1])                       
            Hy[x,y] = Hy[x,y] + 0.5 * (Ez[x+1,y] - Ez[x,y])
    return Hx, Hy

def H_update_slice(Hx, Hy, Ez):
    Hx[:-1, :-1] = Hx[:-1, :-1] + 0.5 * (Ez[:-1,:-1] - Ez[:-1, 1:])                       
    Hy[:-1, :-1] = Hy[:-1, :-1] + 0.5 * (Ez[1:,:-1] - Ez[:-1, :-1])
    return Hx, Hy

# Basic Geometry and Dielectric Parameters
n_xpts = n_ypts = 100  # no of FDTD cells in x and y (dealing with a square region here)
nsteps = 100 # total number of FDTD time steps

# Parallelization
chunk = n_xpts // size
x_offset = rank * chunk
ny = n_ypts
nx = chunk + 1
if rank == size - 1:  # if problem is not divisible by ntasks then modify last slice
    nx = n_xpts - x_offset + 1

# For animation updates - will slow down the loop to see Ex frames better
save_plot = True  # save graph to pdf
cycle = nsteps // 5 - 1  # for graph saving updates
if save_plot: 
    fig = plt.figure(figsize=(8,6))

grid_shape = (nx, ny)
ga = np.ones(grid_shape, dtype=np.float64)

# two time-dependent field monitors
EzMonTime1 = np.empty(nsteps)
PulseMonTime = np.empty(nsteps)

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

# simple fixed dielectric box coordinates
X1 = isource + 10
X2 = X1 + 40
Y1 = jsource + 10
Y2 = Y1 + 40
ga[X1:X2, Y1:Y2] = 1 / eps_box
ga = ga[x_offset : x_offset+nx]  # select only the relevant slice


use_slicing = False
use_jit = True

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


if rank == 0:
    start = timeit.default_timer()      
   
### MAIN LOOP ###
for t in range(nsteps):
    # Communication
    if rank != 0:
        comm.Recv(Hy[0], dest=rank-1, tag=11)
        comm.Send(Ez[0], dest=rank-1, tag=12)
        
    if rank != size-1:
        comm.Send(Hy[-1], dest=rank+1, tag=11)
        comm.Recv(Ez[-1], dest=rank+1, tag=12)

    # iterate pulse (t is an integer, so dt steps)
    pulse = pulse_fn(t)

    # calculate Dz (Hy is diff sign to before with Dz term from curl eqs)
    Dz, Ez = DE_update_fn(Dz, Hy, Hx, Ez, ga)

    Dz[isource, jsource] = Dz[isource, jsource] + pulse  # soft source in simulation center
    Ez[isource, jsource] = ga[isource, jsource] * Dz[isource, jsource]

    # save one point in time just to see the transient
    EzMonTime1[t] = Ez[isource, jsource]
    PulseMonTime[t] = pulse 

    # update H (could also do slicing - but let's make it clear just now)    
    Hx, Hy = H_update_fn(Hx, Hy, Ez)

    if save_plot and t+1 % cycle == 0: # simple animation
        full_Ez = None
        if rank == 0:
            full_Ez = np.empty((n_xpts, n_ypts))
        comm.Gather(Ez[:-1], full_Ez, root=0)

        if rank == 0:
            graph(t, full_Ez, 'p3d')

if rank == 0:
    stop = timeit.default_timer()
    print(f"Time for FDTD simulation: {round(stop - start, 3)} s \n")

# !mpiexec -n 2 python C:\Users\mattr\OneDrive\Documents\University\5th Year\ENPH 479\ps5\2d_parallel_FDTD.py
