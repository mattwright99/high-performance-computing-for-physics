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

def graph(t, Ez, Ez_mon_time, pule_mon_time, save_name=''):
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
    ax.set_title(f"Frame: {t}")

    # Small graph to see time development as a single point
    PulseNorm = pule_mon_time[:t+1] * 0.2
    ax2.plot(PulseNorm,'r',linewidth=1.6)
    ax2.plot(Ez_mon_time[:t+1],'b',linewidth=1.6)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_title(r'$E_{in}(t)$')

    plt.draw()
    plt.savefig(f"./{save_name}_{t}.pdf", dpi=800)

# Basic Geometry and Dielectric Parameters
n_xpts = n_ypts = 3008  # no of FDTD cells in x and y (dealing with a square region here)
nsteps = 1000 # total number of FDTD time steps

# Parallelization
chunk = n_xpts // size
ny = n_ypts
# pad with buffer row at bottom and top
nx = chunk + 2
x_offset = rank * chunk

# For animation updates - will slow down the loop to see Ex frames better
save_plot = False  # save graph to pdf
cycle = nsteps // 5  # for graph saving updates
if save_plot and rank == 0: 
    fig = plt.figure(figsize=(8,6))

# two time-dependent field monitors
EzMonTime1 = np.empty(nsteps)
PulseMonTime = np.empty(nsteps)

c = constants.c  # speed of light in vacuum
fs = constants.femto  # 1.e-15 - useful for pulses 
tera = constants.tera  # 1.e12 - used for optical frequencues

dx = 20.e-9 / 2  #  FDTD grid size in space, in SI Units
dt = dx/(2.*c)  # FDTD time step

isource = int(n_ypts/2)  # x position of pulse sourc
jsource = int(n_xpts/2)  # y position of pulse source
source_rank = isource // chunk  # rank of slice that contains the pulse source
if isource == source_rank * chunk:  # move source if on interface
    isource += 3
if rank == source_rank:
    x_source = isource - x_offset + 1   # x position of pulse with buffer offset

spread = 1 * fs/dt  # 2 fs for this example
t0 = 6 * spread
freq_in = 2*np.pi * 200*tera  # incident (angular) frequency
w_scale = freq_in * dt
eps_box = 9  # dielectric box (so 1 is just free space)

# simple fixed dielectric box coordinates
ga = np.ones((n_xpts+2, n_ypts), dtype=np.float64)  # add 2 to x dim becuase of slice padding
X1 = isource + 10
X2 = X1 + 40
Y1 = jsource + 10
Y2 = Y1 + 40
ga[X1:X2, Y1:Y2] = 1 / eps_box
ga = ga[x_offset : x_offset + chunk, :]  # select only the relevant slice

grid_shape = (nx, ny)  # shape of slice
Ez = np.zeros(grid_shape, dtype=np.float64)
Dz = np.zeros(grid_shape, dtype=np.float64)
Hx = np.zeros(grid_shape, dtype=np.float64)
Hy = np.zeros(grid_shape, dtype=np.float64) 

# Pulse function
@njit(fastmath=True, cache=True)
def pulse_fn(t):
    return np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*freq_in*dt))

@njit(fastmath=True, cache=True)
def DE_update(Dz, Hy, Hx, Ez, ga):
    for x in range(1, nx-1): 
        for y in range(1, ny-1):
            Dz[x,y] += 0.5 * (Hy[x,y] - Hy[x-1,y] - Hx[x,y] + Hx[x,y-1]) 
            Ez[x,y] = ga[x-1,y] * Dz[x,y]  # offset to ga since it is not padded
    return Dz, Ez

@njit(fastmath=True, cache=True)
def H_update(Hx, Hy, Ez):
    for x in range(nx-1):  # start from 0 for 1 process case - not used otherwise
        for y in range(ny-1): 
            Hx[x,y] += 0.5 * (Ez[x,y] - Ez[x,y+1])                       
            Hy[x,y] += 0.5 * (Ez[x+1,y] - Ez[x,y])
    return Hx, Hy

### MAIN LOOP ###
if rank == 0:
    start = timeit.default_timer()   

for t in range(nsteps):
    # Communicate most recent Hy for updating E/D
    if rank != 0:
        comm.Recv(Hy[0], source=rank-1, tag=11)
    if rank != size-1:
        comm.Send(Hy[-2], dest=rank+1, tag=11)

    Dz, Ez = DE_update(Dz, Hy, Hx, Ez, ga)

    if rank == source_rank:
        # iterate pulse (t is an integer, so dt steps)
        pulse = pulse_fn(t)

        Dz[x_source, jsource] += pulse  # soft source in simulation center
        Ez[x_source, jsource] = ga[x_source-1, jsource] * Dz[x_source, jsource]

        # save one point in time just to see the transient
        EzMonTime1[t] = Ez[x_source, jsource]
        PulseMonTime[t] = pulse
        
    # Communicate most recent Ez for updating H
    if rank != 0:
        comm.Send(Ez[1], dest=rank-1, tag=12)
    if rank != size-1:
        comm.Recv(Ez[-1], source=rank+1, tag=12)

    Hx, Hy = H_update(Hx, Hy, Ez)

    if save_plot and (t+1) % cycle == 0:
        if size == 1:
            graph(t, Ez, EzMonTime1, PulseMonTime, 'p3d')
        else:
            # Collect all of Ez to plot
            full_Ez = None
            if rank == 0:
                full_Ez = np.empty((n_xpts, n_ypts))
            comm.Gather(Ez[1:-1], full_Ez, root=0)

            # Collect moment in time data for plots in rank 0 as well
            if rank == source_rank:
                comm.Send(EzMonTime1, dest=0, tag=13)
                comm.Send(PulseMonTime, dest=0, tag=14)

            if rank == 0:
                comm.Recv(EzMonTime1, source=source_rank, tag=13)
                comm.Recv(PulseMonTime, source=source_rank, tag=14)
                graph(t, full_Ez, EzMonTime1, PulseMonTime, 'p3d')

if rank == 0:
    stop = timeit.default_timer()
    print(f"Time for FDTD simulation: {round(stop - start, 3)} s \n")

