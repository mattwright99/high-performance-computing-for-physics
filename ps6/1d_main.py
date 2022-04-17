"""



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

def plot_eq(x, arr, an_arr=None, ylabel='', xlabel=''):
    """Plot a quantity and its analytical solution over x"""

    fig = plt.figure(figsize=(8,6))

    ax = fig.add_axes([.2,.2,.6,.6])
    ax.plot(x, arr,  '.')
    if an_arr is not None:
        ax.plot(x, an_arr, 'r-')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(True, linestyle=':')

    plt.show()

def plot_1d_ising(n_steps, n_spins, E_arr, spin_arr, kT):
    """Plot the spin evolution and the energy over iteraion of the 1D Ising model"""

    fig = plt.figure(figsize=(8, 6))

    # Plot spins over time
    ax1 = fig.add_axes([.2,.55,.6,.35])
    ax1.imshow(spin_arr.T, origin='lower', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks([])
    ax1.set_ylabel('N Spins')
    ax1.set_title(f'Spin evolution with $k_BT$ = {kT}')

    # Plot energy over time
    ax2 = fig.add_axes([.2,.2,.6,.3])
    E_an = -n_spins * eps * np.tanh(eps / kT)  # analytical equilibrium energy
    E_scale = 1 / (n_spins * eps)
    ax2.plot(np.arange(0, n_steps, 1/n_spins), E_arr*E_scale, 'b-', label=r'$E$')
    ax2.axhline(y=E_an*E_scale, c='r', label=r'$\langle E \rangle_{an}$')
    ax2.set_ylim(-1.1, 0)
    ax2.set_ylabel(r'Energy$/N\epsilon$')
    ax2.set_xlabel('Iteration/N')
    ax2.legend()

    if animate:
        # Quick and dirty animation of evolution for 50 frame
        for t in range(0, n_steps, n_steps//25):
            # Remake spin state plot
            ax1.clear()
            ax1.imshow(spin_arr[:(t+1)*n_spins].T, origin='lower', aspect='auto', vmin=-1, vmax=1)
            ax1.set_xticks([])
            ax1.set_ylabel('N Spins')
            ax1.set_title(f'Spin Evolution: $k_BT$ = {kT}, Iteration = {t}N')
            
            # Expand energy limits
            ax2.set_xlim(0, t+1)

            plt.draw()
            plt.pause(0.1)

    plt.show()

@njit()
def initialize_1d(N, p):
    """Initialize the 1D Ising model"""

    spin = np.ones(N)
    E = 0  # Energy
    M = 0  # Magnetization
    for i in range(1, N):
        if np.random.rand(1) < p:
            spin[i] = -1
        E -= spin[i - 1] * spin[i]
        M += spin[i]
    # periodic BC inclusion
    E -= spin[N - 1] * spin[0]
    M += spin[0]
    return spin, E, M

@njit()
def update_1d_microstate(N, spin, kT, E, M, p):
    """Choose a random index and see if it flips or not using the Metropolis ALgorithm"""

    num = np.random.randint(0, N)
    flip = 0
    # periodic BC returns 0 if num + 1 == N , else no change :
    dE = 2 * spin[num] * (spin[num - 1] + spin[(num + 1) % N])
    # if dE is negative , accept flip :
    if dE < 0:
        flip = 1
    else:
        p = np.exp(-dE / kT)
    if np.random.rand(1) < p:
        flip = 1
    # otherwise , reject flip
    if flip == 1:
        E += dE
        M -= 2 * spin[num]
        spin[num] = -spin[num]
    return E, M, p

@njit()
def update_1d_spins(i, E_arr, M_arr, spin_arr, spin, E, M, N, kT, p):
    """Call the randomized individual spin update function N times.
    
    Parameters
    ----------
    i : int
        Step number. Each step is N iterations.
    E_arr : numpy.ndarray
        Energy value array to populate with for each iteration. Passed by reference for
        modification.
    M_arr : numpy.ndarray
        Magnetization value array to populate with for each iteration. Passed by reference
        for modification.
    spin_arr : numpy.ndarray
        Spin value array to populate with for each iteration. Passed by reference for
        modification.
    spin : numpy.ndarray
        Spin values for current iteration.
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

    for j in range(N):
        idx = i*N + j
        E_arr[idx] = E
        M_arr[idx] = M
        spin_arr[idx, :] = spin.copy()

        E, M, p = update_1d_microstate(N, spin, kT, E, M, p)
    
    return E, M, p
            
def ising_1d_loop(n_steps, n_spins, p0, kT, plotting=True):
    """Use the Metropolis Algorithm to evolve a 1D Ising model.
    
    Parameters
    ----------
    n_steps : int
        Number of "time steps" to evolve the system for. Each time step calls the
        `update_1d_microstate` function N times to randomly see if any spins swap.
    n_spins : int
        Number of spins, N.
    p0 : float
        Initial order parameter.
    kT : float
        Defines the temperature of the bath scaled by the Boltzman constant and in units of
        epsilon which we take to be 1.
    plotting : bool
        Flag to determine whether or not to plot evolution of the system. Also determines
        whether or not to print execution time of loop.

    Returns
    -------
    E_arr : numpy.ndarray
        Energy value for every iteration (i.e. size of `(n_spins*n_steps, )`).
    M_arr : numpy.ndarray
        Magnetization value for every iteration (i.e. size of `(n_spins*n_steps, )`).
    """

    # Declare arrays to track system evolution
    spin_arr = np.empty((n_steps*n_spins, n_spins))  # spin states
    E_arr = np.empty(n_steps*n_spins)  # energy
    M_arr = np.empty(n_steps*n_spins)  # magnetization

    spin, E, M = initialize_1d(n_spins, p0)
    p = p0
    ts = timeit.default_timer()
    for i in range(n_steps):
        E, M, p = update_1d_spins(i, E_arr, M_arr, spin_arr, spin, E, M, n_spins, kT, p)
    
    if plotting:
        print(f'Time: {round(timeit.default_timer() - ts, 4)} s')
        plot_1d_ising(n_steps, n_spins, E_arr, spin_arr, kT)

    return E_arr, M_arr

def ising_temp_sweep(kT_vals, n_eq, n_mc, n_spins, p0):
    """Use the Metropolis Algorithm to evolve a 1D Ising model.
    
    Parameters
    ----------
    kT_vals : iterable
        List of temperature values to sweep over.
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
    S_avg : numpy.ndarray
        Average measured equilibirum entropy for given temperature values.
    """

    n_steps = n_eq + n_mc
    # Declare arrays to save expectation values
    E_avg = []
    M_avg = []
    S = 0  # entropy computed by integrating
    S_avg = []

    # Run Ising solver for different temperatures
    ts = timeit.default_timer()
    for i, kT in enumerate(kT_vals):
        E_arr, M_arr = ising_1d_loop(n_steps, n_spins, p0, kT, plotting=False)

        # Compute average at equilibrium
        E_avg.append(np.mean(E_arr[n_eq*n_spins:]))
        M_avg.append(np.mean(M_arr[n_eq*n_spins:]))

        if i == 0:  # start entropy at zero
            S += 0
        else:
            S += (E_avg[i] - E_avg[i-1]) / kT
        S_avg.append(S)
    print(f'Time for simulation: {timeit.default_timer() - ts}')
    return E_avg, M_avg, S_avg

# Declare constants
mu = 1  # permeability
eps = 1  # (epsilon) dielectric constant, permitivity
k_B = constants.Boltzmann  # Boltzman constant

# Set to True if you want to animate the spin/energy evolution instead of just plotting
# the final state
animate = True
if animate:
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'auto')

#%% Question 1 (a) - Visualize solution for different temperatures

print_title('Question 1 (a)')

p0 = 0.6
n_spins = 50
n_steps = 500

for kT in [0.1, 0.5, 1]:
    ising_1d_loop(n_steps, n_spins, p0, kT, plotting=True)


#%% Question 1 (b) - Average equilibrium energy, magnetization, and entropy

print_title('Question 1 (b)')

p0 = 0.6
n_spins = 50
n_steps = 500

# This code was used to estimate how many iterations it takes to reach equilibrium
fig = plt.figure(figsize=(8, 8))
for i, kT in enumerate(np.linspace(1e-3, 6, 6)):
    E_arr, _ = ising_1d_loop(n_steps, n_spins, p0, kT, plotting=False)

    # Add energy plot to subplolts
    plt.subplot(6, 1, i+1)
    
    E_an = -n_spins * eps * np.tanh(eps / kT)  # analytical equilibrium energy
    E_scale = 1 / (n_spins * eps)
    plt.plot(np.arange(0, n_steps, 1/n_spins), E_arr*E_scale, 'b-', label=r'$E$')
    plt.axhline(y=E_an*E_scale, c='r', label=r'$\langle E \rangle_{an}$')
    plt.ylim(-1.1, 0.2)
    plt.xlim(0, n_steps)
    plt.ylabel(r'$E/N\epsilon$')
    if i == 5:
        plt.xlabel('Iteration/N')

plt.tight_layout()
plt.show()

print("""
As we expect, it is clear that the lower temperture cases take longer to reach equilibrium
but are more stable once there. After a few trials, it is seen that the lowest temperature
solution reaches equilibrium within 400N steps. Thus, to compute average equilibrium energy,
magnetization, and entropy, we will begin keeping track of the points after 400N steps.
""")

n_spins=50

# Set iteration numbers
n_eq = 400  # number of steps to reach equilibrium
n_mc = 400  # number of Monte Carlo samples
n_steps = n_eq + n_mc

kT_vals = np.linspace(1e-2, 6, 200)  # temperatures to use
E_avg, M_avg, S_avg = ising_temp_sweep(kT_vals, n_eq, n_mc, n_spins, p0)

kT_vals = kT_vals / eps  # scale for plotting

# Plot energy
E_scale = 1 / (n_spins * eps)
E = E_scale * np.asarray(E_avg)
E_an = -n_spins * eps * np.tanh(eps / kT_vals) * E_scale  # analytical equilibrium energy
plot_eq(kT_vals, E, E_an,
    ylabel=r'$\langle E \rangle / N \epsilon$', xlabel=r'$k_B T/\epsilon$'
)
# Plot magentization
M_scale = 1 / n_spins
M = M_scale * np.asarray(M_avg)
M_an = np.zeros(kT_vals.size)
plot_eq(kT_vals, M, M_an,
    ylabel=r'$\langle M \rangle / N$', xlabel=r'$k_B T/\epsilon$'
)
# Plot entropy
S_scale = 1 / n_spins  # already normalized by k_B
S = S_scale * np.asarray(S_avg)
S_an = S_scale * n_spins * (np.log(2*np.cosh(eps/kT_vals)) - eps/kT_vals*np.tanh(eps/kT_vals))
plot_eq(kT_vals, S, S_an,
    ylabel=r'$\langle S \rangle / Nk_B$', xlabel=r'$k_B T/\epsilon$'
)
