import numpy as np
import matplotlib.pyplot as plt

# general
L = 1 # m, rod length
n = 100 # number of points

# space
x = np.linspace(0, L, n) # evenly distrib points
dx = L/n # delta x

# time
t_final = 61 # s, length of simulation
dt = 0.1 # delta time

# snapshots
snap = [0, 5, 10, 20, 30, 60] # s, seconds at which to take a snapshot

# initial function
func = lambda x: 20 + ( 30 * np.exp(-100 * (-0.5 + x)**2) )
T = np.zeros(n) # init temperature vector
for i, x_ in enumerate(x):
    T[i] = func(x_)

# coefficient
alpha = 2.3e-4 # m^2/s, aluminum diffusion coefficient

dTdt = np.zeros(n) # vector of temperature changes

t_max = max(T)
t_min = min(T)

# Courant Friedrichs Lewy condition to assure that time steps are small enough
cfl = alpha * dt / dx**2
if cfl >= 0.5:
    raise Exception("CFL condition failed. CFL must be less than 0.5. Current value: {}".format(cfl))

# Main program
for j in np.arange(0,t_final,dt):
    for i in range(1,n-1):
        dTdt[i] = ( alpha*dt/dx**2 )*( T[i-1] - 2*T[i] + T[i+1] )
    T = T + dTdt
    if j in snap:
        plt.plot(x,T)
        plt.axis([0, L, t_min, t_max])
        plt.xlabel('Rod length (m)')
        plt.ylabel('Temperature (C)')
        plt.savefig('snapshot{}.png'.format(j), dpi=200)
        plt.clf()