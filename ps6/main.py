"""




"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import timeit
from scipy import constants

# comment these if any problems - sets graphics to auto
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'auto')


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


def plot_1d_ising(n_steps, n_spins, E_arr, spin_arr, kT):

    fig = plt.figure(figsize=(8, 6))

    # Plot spins over time
    ax1 = fig.add_axes([.2,.55,.6,.35])
    ax1.imshow(spin_arr.T, origin='lower', aspect='auto')
    ax1.set_xticks([])
    ax1.set_ylabel('N Spins')
    ax1.set_title(f'Spin evolution with $k_BT$ = {kT}')

    # Plot energy over time
    E_an = -n_spins * eps * np.tanh(eps / kT)  # analytical equilibrium energy

    ax2 = fig.add_axes([.2,.2,.6,.3])
    E_scale = 1 / (n_spins * eps)
    ax2.plot(np.arange(0, n_steps, 1/n_spins), E_arr*E_scale, 'b-', label=r'$E$')
    ax2.axhline(y=E_an*E_scale, c='r', label=r'$\langle E \rangle_{an}$')
    ax2.set_ylim(-1.1, 0)
    ax2.set_xlim(0, n_steps)
    ax2.set_ylabel(r'Energy$/N\epsilon$')
    ax2.set_xlabel('Iteration/N')
    ax2.legend()

    plt.show()

def initialize(N, p):
    """Initialize ... TODO """

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
def update(N, spin, kT, E, M, p):
    """Update ... TODO """

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
def update_spins(i, E_arr, M_arr, spin_arr, spin, E, M, N, kT, p):
    """Call the randomized spin update function N times.
    
    Parameters
    ----------
    
    """

    for j in range(N):
        idx = i*N + j
        E_arr[idx] = E
        M_arr[idx] = M
        spin_arr[idx, :] = spin.copy()

        E, M, p = update(N, spin, kT, E, M, p)
    
    return E, M, p
            
def ising_1d_loop(n_steps, n_spins, p0, kT):
    """Use the Metropolis Algorithm to evolve a 1D Ising model.
    
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

    # Declare arrays to track system evolution
    spin_arr = np.empty((n_steps*n_spins, n_spins))  # spin states
    E_arr = np.empty(n_steps*n_spins)  # energy
    M_arr = np.empty(n_steps*n_spins)  # magnetization

    spin, E, M = initialize(n_spins, p0)
    p = p0
    ts = timeit.default_timer()
    for i in range(n_steps):
        E, M, p = update_spins(i, E_arr, M_arr, spin_arr, spin, E, M, n_spins, kT, p)
    print(f'Time: {round(timeit.default_timer() - ts, 4)} s')

    # plot_1d_ising(n_steps, n_spins, E_arr, spin_arr, kT)

    return E_arr, M_arr


#%% Question 1 (a)

print_title('Question 1 (a)')

p0 = 0.6
n_spins = 500
n_steps = 500
eps = 1

"""
kT = 0.1
ising_1d_loop(n_steps, n_spins, p0, kT)

kT = 0.5
ising_1d_loop(n_steps, n_spins, p0, kT)

kT = 1
ising_1d_loop(n_steps, n_spins, p0, kT)
"""
plt.close('all')

#%% Question 1 (b)

print_title('Question 1 (b)')

p0 = 0.6
n_spins = 50
n_steps = 500
eps = 1
k_B = constants.Boltzmann

fig = plt.figure(figsize=(8, 8))
for i, kT in enumerate(np.linspace(1e-3, 6, 6)):
    E_arr, _ = ising_1d_loop(n_steps, n_spins, p0, kT)

    plt.subplot(6, 1, i+1)
    
    E_an = -n_spins * eps * np.tanh(eps / kT)  # analytical equilibrium energy
    E_scale = 1 / (n_spins * eps)
    plt.plot(np.arange(0, n_steps, 1/n_spins), E_arr*E_scale, 'b-', label=r'$E$')
    plt.axhline(y=E_an*E_scale, c='r', label=r'$\langle E \rangle_{an}$')
    plt.ylim(-1.1, 0.2)
    plt.xlim(0, n_steps)
    plt.ylabel(r'Energy$/N\epsilon$')
    if i == 5:
        plt.xlabel('Iteration/N')

plt.tight_layout()
plt.show()

print("""
As we expect, it is clear that the lower temperture cases take longer to reach equilibrium
but are more stable once there. After a few trials, it is seen that the lowest temperature
solution reaches equilibrium within 20N steps. Thus, to compute average equilibrium energy,
magnetization, and entropy, we will begin keeping track of the points after 20N steps.
""")
plt.close('all')

n_eq = 20  # number of steps to reach equilibrium
n_mc = 400  # number of Monte Carlo samples
n_steps = n_eq + n_mc

for i, kT in np.linspace(1e-3, 6, 200):
    E_arr, M_arr = 


