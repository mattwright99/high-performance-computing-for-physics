# -*- coding: utf-8 -*-
"""
Late edited: March 2022

@author: shugh
"""

# fdtd_V1p0.py - Simple Exampel code for Q1 of PS5(Part A)

"""
1d FDTD Simulation for Ex nd Hy, simple dielectric, simple animation
Units for E -> E*sqrt(exp0/mu0), so E and H are comparible (and same for a plane wave)
No absorbing BC (ABC) - so will bounce off walls
Source (injected at a single space point) goes in both directions

Code is not vectorized, but is kept simple to show you what is going on

Set your graphics to Auto, e.g., within Spyder or
%matplotlib Auto
-- Can also add these two lines (works under windows)
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'auto')
"""

from pickle import FALSE
import numpy as np
import scipy.constants as constants
# comment these if any problems - sets graphics to auto
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'auto')

from matplotlib import cm, colors, rcParams
import matplotlib.pyplot as plt

rcParams['font.size'] = 12
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1


# Plotting config
time_pause = 0.8  # specifies time a frame is displayed in animation
save_plot = False  # 0 will not save graphics 
cycle = 100  # for graph updates
animate_flag = False  # specify whether or not to animate result

# initialize graph, fixed scaling for this first example
def init_plt_1():
    plt.ylim((-0.8, 0.8))
    plt.xlim((0, n_xpts-1))    
    plt.axvline(x=n_xpts//2,color='r')  # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.pause(1)

def init_plt_1cd(d_start, dielectric_end):
    plt.ylim((-0.8, 0.8))
    plt.xlim((0, 600))    
    plt.axvline(x=d_start,color='r')  # Vert line separator
    plt.axvline(x=dielectric_end,color='r')  # Vert line separator
    plt.grid('on')
    ax.set_xlabel('Grid Cells ($z$)')
    ax.set_ylabel('$E_x$')
    plt.pause(1)

def update_plot(i, cycle, Ex):
    if i % cycle != 0:
        return
    im.set_ydata(Ex[1:n_xpts-1])
    ax.set_title("frame time {}".format(i))
    plt.draw()
    plt.pause(time_pause) # sinsible pause to watch animation 

    if save_plot and i % 2*cycle == 0:
        plt.savefig("./figs/cycle_{}.pdf".format(i*100), dpi=800)

def plot_1cd(E_in, E_t, E_r, L, epsilon):
    # Plotting function for part 1 (c) and (d)
    fig = plt.figure(figsize=(8,6))

    # Plot electric fields over time
    ax1 = fig.add_axes([.2, .6, .6, .3])
    ax1.plot(range(n_steps + 1), E_in, 'b-', label=r'$E_{in}$')
    ax1.plot(range(n_steps + 1), E_t, 'r-', label=r'$E_{t}$')
    ax1.plot(range(n_steps + 1), E_r, 'g-', label=r'$E_{r}$')
    ax1.set_xlim(0, 3000)
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
    r, t = get_analytical_soln(L, epsilon, freq)
    T_an = np.abs(t)**2
    R_an = np.abs(r)**2
    # compute transmisison and reflection coefficients vs freq
    T = np.abs(E_t_f/ E_in_f)**2
    R = np.abs(E_r_f)**2 / np.abs(E_in_f)**2
    freq = freq / tera  # scale to THz
    # plot all pretty
    ax2.plot(freq, T, 'r', label=r'$T$')
    ax2.plot(freq, T_an, 'r--', label=r'$T_{an}$')
    ax2.plot(freq, R, 'b', label=r'$R$')
    ax2.plot(freq, R_an, 'b--', label=r'$R_{an}$')
    ax2.plot(freq, T+R,'g' , label='Sum')
    ax2.set_xlim(150, 250)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(r'$T, R$')
    ax2.set_xlabel(r'$\omega /2\pi$ (THz)')
    ax2.legend(loc='lower right')

    # plt.savefig("./figs/p1d.pdf", dpi=800)
    plt.show()

def plot_2(E_in, E_t, E_r, L, epsilon_w, xlims=[0, 300], save_plt=False):
    # Plotting function for part 1 (c) and (d)
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

    if save_plt:
        plt.savefig(f"./figs/pNUM_{int(L*10**9)}nm.pdf", dpi=800)
    plt.show()


def get_analytical_soln(L, eps, f):
    # Analytical solutions for T and R using an assumed harmonic solution
    k0 = 2*np.pi * f / c
    n = np.sqrt(eps)
    r1 = (1-n) / (1+n)
    r2 = (n-1) / (n+1)

    r = (r1 + r2 * np.exp(2j * k0 * L * n)) / (1 + r1*r2*np.exp(2j * k0 * L * n))
    t = (1 + r1) * (1 + r2) * np.exp(1j * k0 * L * n) / (1 + r1*r2*np.exp(2j * k0 * L * n))
    return r, t

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
    ppw = lam // dx  # points per wavelength
    if ppw <= 15:
        raise Exception(f'Points per wavelength should be > 15 but got {ppw}')

isource = 200  # source position
spread = 2 * fs/dt  # 2 fs for this example
t0 = spread * 6
freq_in = 2*np.pi * 200 * tera  # incident (angular) frequency
w_scale = freq_in * dt
check_ppw(1, freq_in, dx)

# Pulse function
pulse_fn = lambda t: -np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*w_scale))

# TODO: obtain numerical reflection of ABC
def a_FDTD_loop_1D(nsteps, cycle):
    if animate_flag:
        init_plt_1()

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
            update_plot(i, cycle, Ex)

def b_FDTD_loop_1D(nsteps, cycle):
    if animate_flag:
        init_plt_1()

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
            update_plot(i, cycle, Ex)

def c_FDTD_loop_1D(nsteps, cycle, L, epsilon):
    d_start = 300  # initial position of dielectric
    d_thickness = int(L / dx)  # thickness in terms of x indices
    permitivity_coeff = np.ones(n_xpts)
    permitivity_coeff[d_start : d_start + d_thickness] = 1 / epsilon

    if animate_flag:
        init_plt_1cd(d_start, d_start+d_thickness)

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
            update_plot(i, cycle, Ex)
    
    return E_in, E_t, E_r

#%%

#  Define first (only in this simple example) graph for updating Ex at varios times
if animate_flag:
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_axes([.18, .18, .7, .7])
    im, = ax.plot(np.arange(1, n_xpts-1), np.zeros(n_xpts-2), linewidth=2)

"Main FDTD: time steps = nsteps, cycle for very simple animation"
# # initialize, then we will just update the y data and title frame
# a_FDTD_loop_1D(n_steps, cycle)
# b_FDTD_loop_1D(n_steps, cycle)

run = False  # whether to run simulation or load saved results
n_steps = 30000  # simulate for a long time to ensure signals die down
epsilon = 9  # permitivity coefficient
L = 1e-6  # length of dielectric
check_ppw(np.sqrt(epsilon), freq_in, dx)
# E_in, E_t, E_r = c_FDTD_loop_1D(n_steps, cycle, L, epsilon)
# plot_1cd(E_in, E_t, E_r, L, epsilon)


#%%  QUESTION 2: Flux density, lossy fre-dependent mediums, Z-transform

def flux_density_FDTD_loop_1D(nsteps, cycle, L, Sx_function):
    """Question 2: Add ability to model dispersion models"""

    d_start = 300  # initial position of dielectric
    d_thickness = int(L / dx)  # thickness in terms of x indices
    d_end = d_start + d_thickness
    
    if animate_flag:
        init_plt_1cd(d_start, d_end)

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


# Update pulse
spread = 1 * fs/dt  # 1 fs for this example
t0 = spread * 6
# check_ppw(np.sqrt(epsilon), freq_in, dx)
pulse_fn = lambda t: -np.exp(-0.5*(t-t0)**2/spread**2)*(np.cos(t*w_scale))  # Pulse function

#%% Question 2 (b)
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

n_steps = 30000  # simulate for a long time to ensure signals die down

# L = 200e-9
# E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_drude_fn)
# plot_2(E_in, E_t, E_r, L, epsilon_D, xlims=[0, 300])

# L = 800e-9
# E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_drude_fn)
# plot_2(E_in, E_t, E_r, L, epsilon_D, xlims=[100, 400])

#%% Question 2 (c)

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

n_steps = 30000  # simulate for a long time to ensure signals die down

# L = 200e-9
# E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_lorentz_fn)
# plot_2(E_in, E_t, E_r, L, epsilon_L, xlims=[150, 250])

# L = 800e-9
# E_in, E_t, E_r = flux_density_FDTD_loop_1D(n_steps, cycle, L, Sx_lorentz_fn)
# plot_2(E_in, E_t, E_r, L, epsilon_L, xlims=[150, 250])


