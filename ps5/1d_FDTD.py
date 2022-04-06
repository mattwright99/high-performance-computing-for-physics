# -*- coding: utf-8 -*-
"""
ENPH 479 PS 5: Part 1 and 2 code: 1-dimensional FDTD

This file holds my code for a 1-dimensional Finite-Difference Time-Domain simulation to
solve Maxwell's equations. 

We begin with 3 varaitions of the same function, `FDTD_loop_1D`, which may be overkill but
they show a clean progression following the steps: (a) implementing absorbing boundary
conditions, (b) implementing total-field scattered-field approach, and (c) injecting pulse
into a dielectric film. Then for Part 2, we implement a flux density variation of this
algorithm to simulate lossy media, namely, the Drude and Lorentz dispersion models using
Z-transformations.
"""

import numpy as np
import scipy.constants as constants
import matplotlib.pyplot as plt

# comment these if any problems - sets graphics to auto
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')


plt.rcParams['font.size'] = 15
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1

# %matplotlib Auto


# Plotting config
time_pause = 0.5  # specifies time a frame is displayed in animation
cycle = 100  # for graph updates
animate_flag = True  # specify whether or not to animate result
save_plot = False  # False will not save graphics (must have animate_flag == True as well)


def init_plt_1D(ybounds, d_start=None, d_end=None):
    """Figure initialization function used for 1D FDTD simulation

    Parameters
    ----------
    ybounds : iter
        Iterable of min, max bounds for y-axis for simulation plotting.
    d_start : int
        Starting index of dielectric film.
    d_end : int
        Ending index of dielectric film.
    """
    
    plt.ylim(*ybounds)
    plt.xlim((0, n_xpts-1))
    if d_start and d_end:  # plot lines showing to dielectric film
        plt.axvline(x=d_start, color='r')  # Vert line separator
        plt.axvline(x=d_end, color='r')  # Vert line separator
    else:  # otherwise plot mid-point
        plt.axvline(x=n_xpts//2,color='r')  # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.pause(1)

def update_plot(i, cycle, Ex, question_name=''):
    """Plotting function used for 1D FDTD simulation animation
    
    Parameters
    ----------
    i : int
        Frame step.
    cycle : int
        Animation update frequency.
    Ex : numpy.ndarray
        Array of scaled electric field values.
    question_name : str
        Name of question to save plot under.
    """
    
    # Check if we update during this step or not
    if i % cycle != 0:
        return
    im.set_ydata(Ex[1:n_xpts-1])
    ax.set_title(f"Frame: {i}")
    plt.draw()
    plt.pause(time_pause) # sinsible pause to watch animation 

    if save_plot and i % 2*cycle == 0:
        plt.savefig(f"./figs/{question_name}_cycle_{i}.pdf", dpi=800)

def summ_plot(E_in, E_t, E_r, L, epsilon_w, xlims=[150, 250], save_name=''):
    """Summary plot function for part 1 (c) and (d) and part 2. Shows Ex over time on upper
    axes and then the frequency domain of the transmission and reflection coefficients as
    well as their analytical solution and sum.

    Parameters
    ----------
    E_in : numpy.ndarray
        Incident E field over simulation time.
    E_t : numpy.ndarray
        Transmitted E field over simulation time.
    E_r : numpy.ndarray
        Refelcted E field over simulation time.
    L : int
        Thickness of dielectric slab.
    epsilon_w : callable
        Frequency (angular) dependent dieletric constant.
    xlims : iter
        Limits (min, max) on the x-axis for the frequency domain plot.
    save_name : str
        If provided, save the plot under this given name.
    """
    
    plt.close('all')
    fig = plt.figure(figsize=(8,6))

    # Plot electric fields over time
    ax1 = fig.add_axes([.2, .6, .6, .3])
    ax1.plot(range(n_steps + 1), E_in, 'b-', label=r'$E_{in}$')
    ax1.plot(range(n_steps + 1), E_t, 'r-', label=r'$E_{t}$')
    ax1.plot(range(n_steps + 1), E_r, 'g-', label=r'$E_{r}$')
    ax1.set_xlim(0, 2000)
    ax1.set_ylabel(r'$E_x$')
    ax1.set_xlabel(r'Time $(\Delta t)$')
    ax1.legend(loc='upper right')

    # Plot reflecred and transmitted in frequency domain
    ax2 = fig.add_axes([.2,.2,.6,.3])
    # find frequency domain reps of E fields
    E_in_f = np.fft.rfft(E_in, norm='ortho')
    E_t_f = np.fft.rfft(E_t, norm='ortho')
    E_r_f = np.fft.rfft(E_r, norm='ortho')
    freq = np.fft.rfftfreq(E_in.size, d=dt)
    
    # get the analytical solution to compare
    omega = 2*np.pi * freq
    epsilon = epsilon_w(omega)
    r, t = get_analytical_soln(L, epsilon, freq)
    T_an = np.abs(t)**2
    R_an = np.abs(r)**2
    
    # compute transmisison and reflection coefficients vs freq
    T = np.abs(E_t_f/ E_in_f)**2
    R = np.abs(E_r_f)**2 / np.abs(E_in_f)**2
    
    # plot all pretty
    freq = freq / tera  # scale to THz
    ax2.plot(freq, T, 'r', label=r'$T$', linewidth=0.8)
    ax2.plot(freq, T_an, 'r--', label=r'$T_{an}$', linewidth=2)
    ax2.plot(freq, R, 'b', label=r'$R$', linewidth=0.8)
    ax2.plot(freq, R_an, 'b--', label=r'$R_{an}$', linewidth=2)
    ax2.plot(freq, T+R,'g' , label='Sum')
    ax2.set_xlim(*xlims)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(r'$T, R$')
    ax2.set_xlabel(r'$\omega /2\pi$ (THz)')
    ax2.legend(loc='lower right')

    if save_name:
        plt.savefig(f"./figs/{save_name}nm.pdf", dpi=800)

    plt.pause(5)  # may disappear if using Sypder and show() so add a brief pause
    plt.show()

def get_analytical_soln(L, eps, f):
    """Determines the analytical values for the tranmission and refelction coefficients
    
    Parameters
    ----------
    L :  int
        Thickness of dielectric.
    eps : numpy.ndarray, int, float
        Dielectric constant over space.
    f : numpy.ndarray
        Angular frequency array.

    Returns
    -------
    r : numpy.ndarray
        Reflection coefficient
    t : numpy.ndarray
        Tranmsission coefficient
    """

    # Analytical solutions for T and R using an assumed harmonic solution
    k0 = 2*np.pi * f / c
    n = np.sqrt(eps)
    r1 = (1-n) / (1+n)
    r2 = (n-1) / (n+1)

    r = (r1 + r2 * np.exp(2j * k0 * L * n)) / (1 + r1*r2*np.exp(2j * k0 * L * n))
    t = (1 + r1) * (1 + r2) * np.exp(1j * k0 * L * n) / (1 + r1*r2*np.exp(2j * k0 * L * n))
    return r, t

def a_FDTD_loop_1D(nsteps, cycle):
    """Simulate FDTD with absorbing boundary conditions"""
    
    if animate_flag:
        init_plt_1D(ybounds=[-0.6, 0.5])

    # Keep track of previous 2 values of E at boundary for absorbing BC
    Ex_1 = {'n-1' : 0,  'n-2' : 0}
    Ex_end = {'n-1' : 0,  'n-2' : 0}

    Ex = np.zeros(n_xpts, dtype=np.float64)  # E array  
    Hy = np.zeros(n_xpts, dtype=np.float64)  # H array
    
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
            update_plot(i, cycle, Ex, 'p1a')

def b_FDTD_loop_1D(nsteps, cycle):
    """Simulate FDTD with forward injected pulse"""
   
    if animate_flag:
        init_plt_1D(ybounds=[-1, 0.8])

    # Keep track of previous 2 values of E at boundary for absorbing BC
    Ex_1 = {'n-1' : 0,  'n-2' : 0}
    Ex_end = {'n-1' : 0,  'n-2' : 0}
    
    Ex = np.zeros(n_xpts, dtype=np.float64)  # E array  
    Hy = np.zeros(n_xpts, dtype=np.float64)  # H array

    # loop over all time steps
    for i in range(nsteps+1):
        t = i-1  # iterative time dep pulse as source
        E_pulse = pulse_fn(t)
        H_pulse = pulse_fn(t + 0.5)

        # Update stored boundary points
        Ex_1['n-2'] = Ex_1['n-1']
        Ex_1['n-1'] = Ex[1]
        Ex_end['n-2'] = Ex_end['n-1']
        Ex_end['n-1'] = Ex[-2]

        # Update E
        Ex[1:-1] = Ex[1:-1] + 0.5 * (Hy[0:-2] - Hy[1:-1])
        Ex[isource] = Ex[isource] - 0.5 * H_pulse

        # Apply absorbing BCs
        Ex[0] = Ex_1['n-2']
        Ex[-1] = Ex_end['n-2']
                
        # Update H - note the offset
        Hy[0:-1] = Hy[0:-1] + 0.5 * (Ex[0:-1] - Ex[1:])
        Hy[isource - 1] = Hy[isource - 1] - 0.5 * E_pulse

        # Update graph every cycle
        if animate_flag:
            update_plot(i, cycle, Ex, 'p1b')

def cd_FDTD_loop_1D(nsteps, cycle, L, epsilon):
    """Simulate FDTD with dielectric film"""

    d_start = isource+100  # initial position of dielectric
    d_thickness = int(L / dx)  # thickness in terms of x indices
    d_end = d_start + d_thickness

    permitivity_coeff = np.ones(n_xpts)
    permitivity_coeff[d_start : d_end] = 1 / epsilon

    if animate_flag:
        init_plt_1D(ybounds=(-1, 1), d_start=d_start, d_end=d_end)

    E_in = np.empty(n_steps+1)  # incident field over time
    E_t = np.empty(n_steps+1)  # transmitted field over time
    E_r = np.empty(n_steps+1)  # refelcted field over time

    # Keep track of previous 2 values of E at boundary for absorbing BC
    Ex_1 = {'n-1' : 0,  'n-2' : 0}
    Ex_end = {'n-1' : 0,  'n-2' : 0}

    Ex = np.zeros(n_xpts, dtype=np.float64)  # E array  
    Hy = np.zeros(n_xpts, dtype=np.float64)  # H array

    # loop over all time steps
    for i in range(nsteps+1):
        t = i-1  # iterative time dep pulse as source
        E_pulse = pulse_fn(t)
        H_pulse = pulse_fn(t + 0.5)

        E_in[i] = pulse_fn(t)
        E_t[i] = Ex[d_start + d_thickness + 10]
        E_r[i] = Ex[isource - 10]

        # Update stored boundary points
        Ex_1['n-2'] = Ex_1['n-1']
        Ex_1['n-1'] = Ex[1]
        Ex_end['n-2'] = Ex_end['n-1']
        Ex_end['n-1'] = Ex[-2]

        # Update E
        Ex[1:-1] = Ex[1:-1] + 0.5 * permitivity_coeff[1:-1] * (Hy[0:-2] - Hy[1:-1])
        Ex[isource] = Ex[isource] - 0.5 * H_pulse

        # Apply absorbing BCs
        Ex[0] = Ex_1['n-2']
        Ex[-1] = Ex_end['n-2']

        # Update H - note the offset
        Hy[0:-1] = Hy[0:-1] + 0.5 * (Ex[0:-1] - Ex[1:])
        Hy[isource - 1] = Hy[isource - 1] - 0.5 * E_pulse

        # Update graph every cycle
        if animate_flag:
            update_plot(i, cycle, Ex, 'p1cd')
    
    return E_in, E_t, E_r

def flux_density_FDTD_loop_1D(nsteps, cycle, L, Sx_function):
    """Simulate FDTD with ability to model dispersion models specified by `Sx_function`"""

    d_start = 300  # initial position of dielectric
    d_thickness = int(L / dx)  # thickness in terms of x indices
    d_end = d_start + d_thickness
    
    if animate_flag:
        init_plt_1D(ybounds=(-1, 1), d_start=d_start, d_end=d_end)

    E_in = np.empty(n_steps+1)  # incident field over time
    E_t = np.empty(n_steps+1)  # transmitted field over time
    E_r = np.empty(n_steps+1)  # refelcted field over time

    # Keep track of previous 2 values of E at boundary for absorbing BC
    Dx_1 = {'n-1' : 0,  'n-2' : 0}
    Dx_end = {'n-1' : 0,  'n-2' : 0}

    Ex = np.zeros(n_xpts, dtype=np.float64)  # E array  
    Hy = np.zeros(n_xpts, dtype=np.float64)  # H array
    Dx = np.zeros(n_xpts, dtype=np.float64)  # D array
    # S array for frequency dependent D (track previous 2)
    Sx_prev = {
        'n-1' : np.zeros(d_thickness, dtype=np.float64),
        'n-2' : np.zeros(d_thickness, dtype=np.float64)
    }
    # loop over all time steps
    for i in range(nsteps+1):
        t = i-1  # iterative time dep pulse as source
        E_pulse = pulse_fn(t)
        H_pulse = pulse_fn(t + 0.5)  # add dt/2 offset (in units of dt)

        # Track indicent, transmitted, and reflected E
        E_in[i] = pulse_fn(t)
        E_t[i] = Ex[d_start + d_thickness + 10]
        E_r[i] = Ex[isource - 10]

        # Update stored boundary points
        Dx_1['n-2'] = Dx_1['n-1']
        Dx_1['n-1'] = Dx[1]
        Dx_end['n-2'] = Dx_end['n-1']
        Dx_end['n-1'] = Dx[-2]

        # Update D accoding to central differencing maxwell eqs
        Dx[1:-1] = Dx[1:-1] + 0.5 * (Hy[0:-2] - Hy[1:-1])
        Dx[isource] = Dx[isource] - 0.5 * H_pulse

        # Apply absorbing BCs
        Dx[0] = Dx_1['n-2']
        Dx[-1] = Dx_end['n-2']

        # Update E
        Sx = Sx_function(Sx_prev, Ex[d_start:d_end])  # compute freq-dependent flux density (use prev Ex)
        Ex = 1 * Dx  # n=1 outside of the dielectric
        Ex[d_start : d_end] = Dx[d_start : d_end] - Sx  # subract dielectric component
        
        # update previous two S values for next iteration
        Sx_prev['n-2'] = Sx_prev['n-1']
        Sx_prev['n-1'] = Sx

        # Update H - note the offset relative to E&D
        Hy[0:-1] = Hy[0:-1] + 0.5 * (Ex[0:-1] - Ex[1:])
        Hy[isource - 1] = Hy[isource - 1] - 0.5 * E_pulse

        # Update graph every cycle
        if animate_flag:
            update_plot(i, cycle, Ex)
    
    return E_in, E_t, E_r


# Basic geometry and dielectric parameters
n_xpts = 801  # number of FDTD cells in x
n_steps = 2000  # number of FDTD tiem steps
c = constants.c  # speed of light in vacuum
fs = constants.femto  # 1.e-15 - useful for pulses 
tera = constants.tera  # 1.e12 - used for optical frequencues 
epsilon_0 = constants.epsilon_0
dx = 20e-9 #  FDTD grid size in space, in SI Units
dt = dx / (2 * c) # FDTD time step

# Pulse parameters and points per wavelength
def check_ppw(n, freq_in, dx):
    # Function checks that there are enough points per wavelength for a given material
    v = c / n  # speed of EM wave in medium
    lam =  2*np.pi * v / freq_in  # wavelength
    ppw = lam / dx  # points per wavelength
    if ppw <= 20:
        raise Exception(f'Points per wavelength should be > 15 but got {ppw}')

isource = 200  # source position
spread = 2 * fs/dt  # 2 fs for this example
t0 = spread * 6
freq_in = 2*np.pi * 200 * tera  # incident (angular) frequency
w_scale = freq_in * dt
check_ppw(1, freq_in, dx)

# Pulse function
pulse_fn = lambda t: -np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*w_scale))

#%% QUESTION 1: 1d FDTD, ABC, total-field scattered-field, reflection and transmission from a simple dielectric.

animate_flag = True  # specify whether or not to animate result
save_plot = False  # False will not save graphics (must have animate_flag == True as well)


#  Define first (only in this simple example) graph for updating Ex at varios times
if animate_flag:
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)

n_steps = 200  # number of FDTD tiem steps

# Main FDTD: time steps = nsteps, cycle for very simple animation
# initialize, then we will just update the y data and title frame
a_FDTD_loop_1D(n_steps, cycle)
b_FDTD_loop_1D(n_steps, cycle)


# Part (c) and (d)
isource = 100
n_xpts = 400
n_steps = 3000  # simulate for longer time (3e4) to ensure signals die down
epsilon = 9  # permitivity coefficient
L = 1e-6  # length of dielectric
check_ppw(np.sqrt(epsilon), freq_in, dx)

if animate_flag:
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)

E_in, E_t, E_r = cd_FDTD_loop_1D(n_steps, cycle, L, epsilon)
summ_plot(E_in, E_t, E_r, L, epsilon_w=lambda w: 9)  # epsilon function is a constant


#%%  QUESTION 2: Flux density, lossy fre-dependent mediums, Z-transform

animate_flag = True  # specify whether or not to animate result
save_plot = False  # False will not save graphics (must have animate_flag == True as well)

isource = 100
n_xpts = 400
    
# Update pulse
spread = 1 * fs/dt  # 1 fs for this example
t0 = spread * 6
pulse_fn = lambda t: -np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*w_scale))  # Pulse function

## Question 2(b)
omega_p = 1.26e15
alpha = 1.4e14

def Sx_drude_fn(Sx_prev, Ex):
    """Compute current Sx (a vector subracted from Dx that expresses frequency dependent
    permitivity) according to the Drude dispersion model in the Z domain."""

    Sx = (1 + np.exp(-alpha * dt)) * Sx_prev['n-1'] \
        - np.exp(-alpha*dt) * Sx_prev['n-2'] \
        + dt * omega_p**2 / alpha * (1 - np.exp(-alpha * dt)) * Ex
    return Sx

# Drude dielectric constant response funcion
epsilon_D = lambda omega: 1 - omega_p**2 / (omega**2 + 1j * omega * alpha)

n_steps = 2000  # simulate for a long time to ensure signals die down


if animate_flag:
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)
    
L = 200e-9
E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_drude_fn)
summ_plot(E_in, E_t, E_r, L, epsilon_D, xlims=[0, 300], save_name='p2b_200')


if animate_flag:
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)
    
L = 800e-9
E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_drude_fn)
summ_plot(E_in, E_t, E_r, L, epsilon_D, xlims=[100, 400], save_name='p2b_800')


## Question 2(c)
alpha = 4*np.pi * tera
omega_0 = 2*np.pi * 200 * tera
f_0 = 0.05
beta = np.sqrt(omega_0**2 - alpha**2)

def Sx_lorentz_fn(Sx_prev, Ex):
    """Compute current Sx (a vector subracted from Dx that expresses frequency dependent
    permitivity) according to the Lorentz dispersion model in the Z domain"""

    Sx = 2 * np.exp(-alpha * dt) * np.cos(beta * dt) * Sx_prev['n-1'] \
        - np.exp(-2 * alpha * dt) * Sx_prev['n-2'] \
        + dt * f_0 * (alpha**2/beta + beta) * np.exp(-alpha * dt) * np.sin(beta * dt) * Ex
    return Sx

# Lorentz dielectric constant response funcion
epsilon_L = lambda omega: 1 + f_0 * omega_0**2 / (omega_0**2 - omega**2 - 2j * omega * alpha)

n_steps = 3000  # simulate for a long time to ensure signals die down

if animate_flag:
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)
    
L = 200e-9
E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_lorentz_fn)
summ_plot(E_in, E_t, E_r, L, epsilon_L, xlims=[150, 250], save_name='p2c_200')


if animate_flag:
    plt.close('all')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)
L = 800e-9
E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_lorentz_fn)
summ_plot(E_in, E_t, E_r, L, epsilon_L, xlims=[150, 250], save_name='p2c_800')


