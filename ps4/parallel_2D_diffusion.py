# -*- coding: utf-8 -*-
"""
Parallelized 2D Heat Equation

This file contain my simulation code for Problem Set 4 of ENPH 479.

This script uses MPI to solve the 2D heat equation using parallel computing.
Parallelization is done for both inititialization and the evolution of the system. To run
the file on 4 processors, one needs to submit the command:
```
!mpiexec -n 4 python parallel_2D_diffusion.py
```
assuming they have ``mpi4py`` installed. It can also be run using a scheduler like SLURM by
sibmitting a batch job using the ``mpi-job.sh`` shell script and the command:
```
sbatch mpi-job.sh
```
Alternatively, the speed-up of assiging more hardware to the problem can be tested by
running the ``timing-job.sh`` shell script in your terminal.


Created on Mon Mar 7

@author: Matt Wright
"""

import timeit
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

PLOTTING = False  # flag for saving and plotting


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

w = h = 20.48  # plate size, mm
print(f"[P{rank}] plate size: {w}x{h} mm")

nx = ny = 1024  # grid size
dx, dy = w/nx, h/ny
print(f"[P{rank}] grid size:{nx}x{ny}")

# parallel parameters
chunk = int(ny / size)  # parallelize along y axis
y_idx_offset = rank * chunk  # offset for initialization

# Thermal diffusivity of steel, mm^2/s
D = 4.2
print(f"[P{rank}] thermal diffusivity: {D}")

# time
nsteps = 1001
dt = dx**2 * dy**2 / (2 * D * (dx**2 + dy**2))
print(f"[P{rank}] dt: {dt}")

plot_ts = np.arange(0, nsteps, 100, dtype=int)  # ploting times

# Initialization - circle osf radius r centred at (cx,cy) (mm)
t_start = timeit.default_timer()  # timer for parallelized code
u = np.zeros((chunk+2, nx), dtype=np.float64)  # add 2 rows for communicated results

T_hot = 2000
T_cool = 300
r = 5.12
cx, cy = w / 2, h / 2
for i in range(nx):
    for j in range(chunk):
        row_i = j + 1 # row index offset by 1 for padding
        p2 = (i*dx - cx)**2 + ((j+y_idx_offset)*dy - cy)**2
        if p2 < r**2:
            radius = np.sqrt(p2)
            u[row_i, i] = T_hot * np.cos(4*radius)**4
        else:
            u[row_i, i] = T_cool


def evolve_2d_diff_eq(u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + D * dt * (
          (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
        + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2)
    return u

t_comm = 0  # communication time (running sum)
t_comp = 0  # computation time (running sum)
for ts in range(nsteps):
    # communication
    t_temp = timeit.default_timer()
    if rank != 0:
        comm.Send(u[1], dest=rank-1, tag=11)
        comm.Recv(u[0], source=rank-1, tag=12)
        
    if rank != size-1:
        comm.Recv(u[-1], source=rank+1, tag=11)
        comm.Send(u[-2], dest=rank+1, tag=12)
    t_comm += timeit.default_timer() - t_temp

    t_temp = timeit.default_timer()
    u = evolve_2d_diff_eq(u)
    t_comp += timeit.default_timer() - t_temp

    if PLOTTING and ts in plot_ts:
        # Communicate current total state to save snapshot
        full_u = None
        if rank == 0:
            full_u= np.empty((nx, ny))
        comm.Gather(u[1 : chunk+1], full_u, root=0)
        
        if rank == 0:  # save and plot full U
            # np.save(f'2d_u_{ts}.npy', full_u)

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_axes([0.2,.2,.6,.6])

            im = ax.imshow(full_u, cmap=plt.get_cmap('hot'), vmin=T_cool,vmax=T_hot)
            ax.set_title('{:.1f} ms'.format((ts+1)*dt*1000))
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')

            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
            cbar_ax.set_xlabel('K', labelpad=20)
            fig.colorbar(im, cax=cbar_ax)
            
            # plt.savefig("./figs/2d_iter_{}.png".format(ts), dpi=100)
            plt.clf()

if rank == 0:  # only display time for root node
    t_total = timeit.default_timer() - t_start
    print(f'[P0]: TOTAL PARALLEL EXECUTION TIME: {t_total}')
    print(f'[P0]: COMMUNICATION: {t_comm}')
    print(f'[P0]: COMPUTATION: {t_comp}')



