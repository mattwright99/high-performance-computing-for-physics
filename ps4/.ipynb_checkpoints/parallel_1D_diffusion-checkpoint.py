import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# general
L = 1 # m, rod length
n = 256 # number of points

# parallel params
chunk = int(n/size)

i_start = 0
if rank == 0:
    i_start = 1 # implicit boundary condition

i_stop = chunk
if rank == size - 1:
    i_stop = chunk - 1 # implicit boundary condition

# space
full_x = None
if rank == 0:
    full_x = np.linspace(0, L, n) # evenly distrib points
x = np.empty(chunk)
comm.Scatter(full_x, x, root=0)

dx = L/n # delta x

# time
t_final = 61 # s, length of simulation
dt = 0.01 # delta time

# snapshots
snap = [0, 5, 10, 20, 30, 60] # s, seconds at which to take a snapshot

# initial function
func = lambda x: 20 + ( 30 * np.exp(-100 * (-0.5 + x)**2) )
T = np.zeros(chunk) # init temperature vector
for i, x_ in enumerate(x):
    T[i] = func(x_)
    
# coefficient
alpha = 2.3e-4 # m^2/s, aluminum diffusion coefficient

dTdt = np.zeros(chunk) # vector of temperature changes

# Courant Friedrichs Lewy condition to assure that time steps are small enough
if rank == 0:
    cfl = alpha * dt / dx**2
    if cfl >= 0.5:
        raise Exception("CFL condition failed. CFL must be less than 0.5. Current value: {}".format(cfl))

# Main program
for j in np.arange(0,t_final,dt):
    # communication
    if rank != 0:
        comm.send(T[0], dest=rank-1, tag=11)
        left_nb = comm.recv(source=rank-1, tag=12)
    
    if rank != size-1:
        comm.send(T[-1], dest=rank+1, tag=12)
        right_nb = comm.recv(source=rank+1, tag=11)
    
    # calculation
    for i in range(i_start,i_stop):
        
        if i == 0:
            left = left_nb
        else:
            left = T[i-1]
            
        if i == chunk-1:
            right = right_nb
        else:
            right = T[i+1]
            
        dTdt[i] = ( alpha*dt/dx**2 )*( left - 2*T[i] + right )
    T = T + dTdt

    # plotting
    if j in snap:
        
        full_T = None
        if rank == 0:
            full_T = np.empty(n)
        comm.Gather(T, full_T, root=0)
        
        if rank == 0:
            plt.plot(full_x,full_T)
            plt.axis([0, L, 20, 50])
            plt.xlabel('Rod length (m)')
            plt.ylabel('Temperature (C)')
            plt.savefig('snapshot{}.png'.format(j), dpi=200)
            plt.clf()