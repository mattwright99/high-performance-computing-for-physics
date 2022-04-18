"""
ENPH 479 PS 6 -- Part 2 code: 2-dimensional Ising Model

This file holds my code for a 2-dimensional Ising model simulation using the Metropolis
Algorithm.

Similar to the 1D script, the first set cell declares the functions required for the
following experiments. We begin by visualizing the system's evolution at a few different
temperatures. Then we execute on a larger grid. Finally, we analyze the equilibrium energy
and magnetization as a function of temperature.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import timeit
from scipy import constants


plt.rcParams['font.size'] = 15
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1

np.random.seed(123)


def print_title(msg):
    """Helper function to print question separations"""

    msg = f' {msg} '
    n = (92 - len(msg)) // 2
    msg = '='*n + msg + '='*n
    print('\n' + msg)

def subplot_eq(kT, E, M):
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

    plt.show()

def plot_2d_ising(n_spins, spin_snaps, kT, snaps, energies):
    """Plot the spin evolution and the energy over iteraion of the 1D Ising model"""

    # Plot spins over time
    fig = plt.figure(figsize=(8, 6))

    for i in range(len(spin_snaps)):
        plt.subplot(len(snaps)//2, 2, i+1)

        plt.imshow(spin_snaps[i], origin='lower')
        plt.title(f'Iteration: {snaps[i]}')

        if i in [0, 2]:
            plt.ylabel('Grid Cells ($y$)')
        if i in [2, 3]:
            plt.xlabel('Grid Cells ($x$)')
    plt.tight_layout()

    # Plot energy over time
    fig = plt.figure(figsize=(8,6))

    ax = fig.add_axes([.2,.2,.6,.6])
    ax.plot(range(len(energies)), energies/n_spins**2, 'b-')
    ax.set_xlabel(r'Iterations/$N^2$')
    ax.set_ylabel(r'$E/N^2 \epsilon$')
    ax.set_title(f'$K_B T = {kT}$')
    ax.set_xlim(0, 100)

    plt.show()

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
    E_arr = np.empty(n_steps + 1)
    M_arr = np.empty(n_steps + 1)

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

def ising_2d_loop(n_steps, n_spins, p0, kT, snap_times=[], n_snaps=None, plotting=True):
    """Use the Metropolis Algorithm to evolve a 2D Ising model. The function,
    `efficient_ising_2d_loop` is equivalent but optimized for computational speed while
    this function is meant for visualization.
    
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
    E_arr = []
    M_arr = []

    # Declare arrays to save snapshots of system evolution
    spin_snapshots = []
    # If snapshot times not specified, save a specified number of snaps
    if not snap_times and n_snaps is not None:
        snap_times = np.linspace(0, n_steps * n_spins**2, n_snaps).astype(int)

    # Randomly initialize spin grid
    spins, E, M = initialize_2d(n_spins, p0)
    E_arr.append(E)
    M_arr.append(M)

    p = p0
    ts = timeit.default_timer()
    for i in range(n_steps):
        # Check to save snap shot of spin state
        iteration = i * n_spins**2
        if iteration in snap_times:
            spin_snapshots.append(spins.copy())

        # Evolve system
        E, M, p = update_2d_spins(spins, E, M, n_spins, kT, p)

        E_arr.append(E)
        M_arr.append(M)

    E_arr = np.asarray(E_arr)
    M_arr = np.asarray(M_arr)

    if plotting:
        print(f'Time: {round(timeit.default_timer() - ts, 4)} s')
        plot_2d_ising(n_spins, spin_snapshots, kT, snap_times, E_arr)

    return E_arr, M_arr

def ising_temp_sweep(kT_vals, n_eq, n_mc, n_spins, p0, ising_loop_fn):
    """Use the Metropolis Algorithm to evolve a 2D Ising model over given temperatures.
    
    Parameters
    ----------
    max_kT : float
    n_eq : int
        Number of "time steps" to evolve the system to equilibium. Equilibrium averaging
        begins after this iteration. Each time step calls the `update_1d_microstate`
        function N times.
    n_mc : int
        Number of "time steps" to evolve the system in equilibirium for Monte Carlo
        averaging. Each time step calls the `update_1d_microstate`
        function N times.
    n_spins : int
        Number of spins, N.
    p0 : float
        Initial order parameter.

    Returns
    -------
    E_avg : numpy.ndarray
        Average measured equilibirum energy for given temperature values.
    M_avg : numpy.ndarray
        Average measured equilibirum magenetization for given temperature values.
    """

    n_steps = n_eq + n_mc
    # Declare arrays to save expectation values
    E_avg = np.empty(len(kT_vals))
    M_avg = np.empty(len(kT_vals))

    # Run Ising solver for different temperatures
    ts = timeit.default_timer()
    for i, kT in enumerate(kT_vals):
        E_arr, M_arr = ising_loop_fn(n_steps, n_spins, p0, kT)
        # Compute average at equilibrium
        E_avg[i] = np.mean(E_arr[n_eq:])
        M_avg[i] = np.mean(M_arr[n_eq:])
    print(f'Time for simulation: {timeit.default_timer() - ts}')
    return E_avg, M_avg

# Declare constants
mu = 1  # permeability
eps = 1  # (epsilon) dielectric constant, permitivity
k_B = constants.Boltzmann  # Boltzman constant


#%% Question 2: 2-dimensional Ising model
# Visualize solution at different temperatures
 
print_title('Question 2 Visualization')

p0 = 0.6
n_spins = 20
n_steps = 400
snapshot_times = [0, 400, 4000, 16000]

for kT in [0.5, 1.5, 3]:
    ising_2d_loop(n_steps, n_spins, p0, kT, snap_times=snapshot_times, plotting=True)


#%% Run for 40x40 cell grid

print_title('Question 2 40x40')

kT = .5
p0 = 0.6
n_spins = 40
n_steps = n_spins**2
snapshot_times = [0, 1600, 16000, 64000]
ising_2d_loop(n_steps, n_spins, p0, kT, snap_times=snapshot_times, plotting=True)

#%% Plot equilibrium energy and mangetization as a function of temperature

print_title('Question 2 Equilibrium')

n_spins = 20
n_eq = n_spins**2
n_mc = n_spins**2

kT_c = 2 / np.log(1 + np.sqrt(2))  # critical temperature

kT_vals = np.linspace(1e-2, 3*kT_c, 100)
E_avg, M_avg = ising_temp_sweep(kT_vals, n_eq, n_mc, n_spins, p0, efficient_ising_2d_loop)

## The commented line below runs the same operations as above but is slower for large-scale simulation
# E_avg, M_avg = ising_temp_sweep(kT_vals, n_eq, n_mc, n_spins, p0, ising_2d_loop)

# Scale quantities appropriately and plot
E_scale = 1 / (n_spins**2 * eps)
E_avg = E_scale * np.asarray(E_avg)
M_scale = 1 / n_spins**2
M_avg = M_scale * np.asarray(M_avg)
kT_scale = 1 / (kT_c * eps)
kT_vals = kT_scale * kT_vals

subplot_eq(kT_vals, E_avg, M_avg)

