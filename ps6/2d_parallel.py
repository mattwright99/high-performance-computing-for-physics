"""
ENPH 479 PS 6 -- 2-dimensional Parallelized Ising Model

This file holds my code for a parallelized Monte Carlo 2-dimensional Ising model simulation
using the Metropolis Algorithm to analyze energy and magnetization at different
temperatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import timeit
from scipy import constants
from mpi4py import MPI


plt.rcParams['font.size'] = 15
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1


def subplot_eq(kT, E, M, save_name='E_M_result'):
    """Plot energy and magenetization at equilibrium over temperatures"""

    fig = plt.figure(figsize=(8,6))

    ax = fig.add_axes([.15,.2,.35,.6])
    ax.plot(kT, E,  '.')
    ax.set_ylabel(r'$\langle E \rangle / N^2 \epsilon$')
    ax.set_xlabel(r'$k_B T/\epsilon T_c$')
    ax.grid(True, linestyle=':')

    ax = fig.add_axes([.6,.2,.35,.6])
    ax.plot(kT, np.abs(M),  '.')
    ax.set_ylabel(r'$\vert\langle M \rangle\vert / N^2 \mu$')
    ax.set_xlabel(r'$k_B T/\epsilon T_c$')
    ax.grid(True, linestyle=':')

    plt.draw()
    plt.savefig(f"{save_name}.pdf", dpi=800)

@njit()
def initialize_2d(N, p0):
    """Initialize the 2D Ising model"""

    spin = np.ones((N, N))
    E = 0
    M = 0
    for i in range(N):
        for j in range(N):
            if np.random.rand(1) < p0:
                spin[i, j] = -1
            E -= spin[i, j] * (spin[i-1, j] + spin[i, j-1])
            M += spin[i, j]
    # periodic boundary conditions inclusion
    for i in range(1, N): 
        E -= spin[0, i] * spin[N-1, i] + spin[i, 0] * spin[i, N-1] # add left*right and bottom*top
        M += spin[0, i] + spin[i, 0]
    return spin, E, M

@njit()
def update_2d_microstate(N, spins, kT, E, M, p):
    """Choose a random index and see if it flips or not using the Metropolis ALgorithm"""

    # Sample random spin index
    r = np.random.randint(0, N)
    c = np.random.randint(0, N)

    flip = 0
    # periodic BC returns 0 if x + 1 == N , else no change :
    dE = 2 * spins[r, c] * (spins[r-1, c] + spins[(r+1)%N, c] + spins[r,c-1] + spins[r,(c+1)%N])
    # if dE is negative , accept flip:
    if dE < 0:
        flip = 1
    else:
        p = np.exp(-dE / kT)
        if np.random.rand(1) < p:
            flip = 1
    # otherwise , reject flip
    if flip == 1:
        E += dE
        M -= 2 * spins[r,c]
        spins[r,c] = -spins[r,c]
    return E, M, p

@njit()
def update_2d_spins(spins, E, M, N, kT, p):
    """Call the randomized individual spin update function N^2 times.
    
    Parameters
    ----------
    spins : numpy.ndarray
        Spin grid for current iteration.
    E : float or int
        Current energy of the system.
    M : float or int
        Current magnetization of the system.
    N : int
        Number of spins.
    kT : float
        Temperature of the system time the Boltzman constant.
    p : float
        Order parameter.

    Returns
    -------
    E : float or int
        Resulting energy of the system.
    M : float or int
        Resulting magnetization of the system.
    p : float
        Order parameter.
    """

    for _ in range(N**2):
        E, M, p = update_2d_microstate(N, spins, kT, E, M, p)
    return E, M, p

@njit()
def efficient_ising_2d_loop(n_steps, n_spins, p0, kT):
    """Use the Metropolis Algorithm to evolve a 2D Ising model. This function can be
    Numba-ified and has no plotting capabilities. When compared to the `ising_2d_loop`
    function using `ising_temp_sweep` and 30 spins, this function ran in 160 s while the
    other ran in 257 s.
    
    Parameters
    ----------
    n_steps : int
        Number of "time steps" or iterations to evolve the system for.
    n_spins : int
        Number of spins, N.
    p0 : float
        Initial order parameter.
    kT : float
        Defines the temperature of the bath scaled by the Boltzman constant and in units of
        epsilon which we take to be 1.
    """

    # Arrays to keep track of energy and magentization over iterations
    E_arr = np.empty(n_steps + 1, dtype=np.float32)
    M_arr = np.empty(n_steps + 1, dtype=np.float32)

    # Randomly initialize spin grid
    spins, E, M = initialize_2d(n_spins, p0)
    E_arr[0] = E
    M_arr[0] = M

    p = p0
    for i in range(n_steps):
        # Evolve system
        E, M, p = update_2d_spins(spins, E, M, n_spins, kT, p)
        E_arr[i] = E
        M_arr[i] = M

    return E_arr, M_arr


# Declare constants
mu = 1  # permeability
eps = 1  # (epsilon) dielectric constant, permitivity
k_B = constants.Boltzmann  # Boltzman constant

# Ising problem parameter definition
p0 = 0.6
n_spins = 20
n_eq = n_spins**2  # steps until equilibrium
n_mc = n_spins**2  # steps for monte carlo averaging
n_steps = n_eq + n_mc
kT_c = 2 / np.log(1 + np.sqrt(2))  # critical temperature
n_kT_vals = 256  # number of temperatures to compute
kT_vals = np.linspace(1e-2, 3*kT_c, n_kT_vals, dtype=np.float32)

# MPI parameter definition
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

chunk = n_kT_vals // size
# Define subset of temps to parallelize over
kT_vals = kT_vals[rank * chunk : (rank+1) * chunk]

# Declare arrays to save expectation values
E_avg = np.empty(chunk, dtype=np.float32)
M_avg = np.empty(chunk, dtype=np.float32)

# Run Ising solver for different temperatures
ts = timeit.default_timer()
for i, kT in enumerate(kT_vals):
    E_arr, M_arr = efficient_ising_2d_loop(n_steps, n_spins, p0, kT)
    # Compute average at equilibrium
    E_avg[i] = np.mean(E_arr[n_eq:])
    M_avg[i] = np.mean(M_arr[n_eq:])

if rank == 0:
    print(f'Time for simulation: {timeit.default_timer() - ts}')

# Collect all of E and M to plot
full_E = None
full_M = None
if rank == 0:
    full_E = np.empty(n_kT_vals, dtype=np.float32)
    full_M = np.empty(n_kT_vals, dtype=np.float32)
comm.Gather(E_avg, full_E, root=0)
comm.Gather(M_avg, full_M, root=0)

# Scale quantities appropriately and plot
E_scale = 1 / (n_spins**2 * eps)
M_scale = 1 / n_spins**2
kT_scale = 1 / (kT_c * eps)
E_avg = E_scale * E_avg
M_avg = M_scale * M_avg
kT_vals = kT_scale * kT_vals

subplot_eq(kT_vals, E_avg, M_avg)

