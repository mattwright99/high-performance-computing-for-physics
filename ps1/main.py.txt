# -*- coding: utf-8 -*-
"""
Extreme Nonlinear Optics - Dynamics of Coupled ODEs

This file has my code for problem set 1 of ENPH 479.

Created on Thu Jan 13 15:47:24 2022

@author: Matt Wright
"""


import matplotlib.pyplot as plt  
import numpy as np 
import timeit 
from scipy.integrate import odeint

plt.rcParams.update({'font.size': 18})


#%% Helper function definition
# NOTE - Assumes all inputs are np arrays

def euler_forward(f, y, t, h):
    '''Simple vectorized Euler ODE Solver (from sample code)'''
    
    k1 = h * f(y, t)          
    y += k1
    return y


def rk4(f, y, t, h):
    '''Vectorized 4th order Runge Kutta'''

    k1 = h * f(y, t)
    k2 = h * f(y + k1/2, t + h/2)
    k3 = h * f(y + k2/2, t + h/2)
    k4 = h * f(y + k3, t + h)
    
    y += (k1 + 2.*k2 + 2.*k3 + k4) / 6.
    return y


# gamma_d: dephasing
# delta_0L: detuning
def derivs_rwa(y, t):
    '''OBE derivatives with RWA approximation'''

    dy = np.zeros(len(y))
    dy[0] = -gamma_d * y[0] + delta_0L * y[1]                                   # Re[u]
    dy[1] = -gamma_d * y[1] - delta_0L * y[0] + rabi_f(t) / 2 * (2.*y[2] - 1.)  # Im[u]
    dy[2] = -1 * rabi_f(t) * y[1]                                               # n_e
    return dy


def derivs(y, t):
    '''Full-wave OBE derivatives'''

    dy = np.zeros(len(y))
    dy[0] = -gamma_d * y[0] + omega_0 * y[1]                                    # Re[u]
    dy[1] = -gamma_d * y[1] - omega_0 * y[0] + full_rabi_f(t) * (2.*y[2] - 1.)  # Im[u]
    dy[2] = -2. * full_rabi_f(t) * y[1]                                         # n_e
    return dy


def full_rabi_f(t):
    '''Full-wave Rabi Field'''
    return Omega_0 * np.exp(-(t - 5)**2) * np.sin(omega_L * (t - 5) + phi)


#%% QUESTION 1
print(f'\n--- Question 1 ---')

# Paramaters for ODEs - simple CW Rabi problem of coherent RWA OBEs
Omega_0 = 2 * np.pi         
rabi_f = lambda t: Omega_0  # Continuous wave Rabi frequency (inverse time units)

# Ignore detuning and dephasing
gamma_d = 0.0
delta_0L = 0.0

dt = 0.01  # toggle between 0.01 and 0.001 to use ~100 or 1000 points per period
print(f'Time step of h = {dt}')
tmax = 5.  # simulate for 5 cycles
# numpy arrays for time and y ODE set
tlist = np.arange(0.0, tmax, dt)  # gauarantees the same step size
npts = len(tlist)

y_euler = np.zeros((npts, 3))
y_rk4 = np.zeros((npts, 3))

yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
y_euler[0,:] = yinit
y_rk4[0,:] = yinit

y_exact = np.sin(rabi_f(tlist) * tlist / 2) ** 2  # analytical solution

# ---- Call ODE Solvers ----
# Euler:
ts = timeit.default_timer()  # start timer for Euler
for i in range(1,npts):   # loop over time
    y_euler[i,:] = euler_forward(derivs_rwa, y_euler[i-1], tlist[i-1], dt) 

t_euler = timeit.default_timer() - ts
print(f"Time for Euler ODE Solver:  {round(t_euler, 4)} s") 

# RK4:
ts = timeit.default_timer()  # start timer for rk4
for i in range(1,npts):   # loop over time
    y_rk4[i,:] = rk4(derivs_rwa, y_rk4[i-1], tlist[i-1], dt) 

t_rk4 = timeit.default_timer() - ts
print(f"Time for RK4 ODE Solver:  {round(t_rk4, 4)} s")  

#%% Graphics: Q1

fig = plt.figure(figsize=(8,6))

ax1 = fig.add_axes([.2, .2, .6, .6])
# plot analytic and numerically obtained population n_e(t)
plt.plot(tlist, y_exact, 'b-', label='Exact Solution')
plt.plot(tlist, y_euler[:,2], 'r--', label='Forward Euler')
plt.plot(tlist, y_rk4[:,2], 'g-.', label='Runge Kutta 4', linewidth=3)
plt.xlabel(r'Normalized time, $t/\tau$')
plt.ylabel(r'$n_{e}$')
ax1.legend(loc='upper left')
# plt.savefig(f'q1-{dt}.pdf', dpi=1200, bbox_inches="tight")

fig = plt.figure(figsize=(8,6))

ax1 = fig.add_axes([.2, .2, .6, .6])
# plot numerical method differences for obtained n_e
plt.plot(tlist, y_exact - y_euler[:,2], 'r--', label='Euler')
plt.plot(tlist, y_exact - y_rk4[:,2], 'g-.', label='RK4')
plt.xlabel(r'Normalized time, $t/\tau$')
plt.ylabel(r'$\Delta n_{e}$')
ax1.legend(loc='upper left')
# plt.savefig(f'q1-diff-{dt}.pdf', dpi=1200, bbox_inches="tight")


#%% Question 2
print(f'\n--- Question 2 ---')

Omega_0 = 2 * np.sqrt(np.pi)  # 2\pi pulse
rabi_f = lambda t: Omega_0 * np.exp(-1 * (t - 5.)**2)  # gaussian pulse with normized time

gamma_d = 0.0
delta_0L = 0.0

dt = 0.01  # time step resolition for ODE solver
tmax = 10.  # simulare for a duration of 10 pulses
print(f'Time step of h = {dt}')
print(f'Duration of tend = {tmax}tp')

# numpy arrays for time and y ODE set
tlist = np.arange(0.0, tmax, dt)
npts = len(tlist)
y_rwa = np.zeros((npts, 3))  # rotating wave approcimation results (will compare with in Q4)

yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
y_rwa[0,:] = yinit

ts = timeit.default_timer()  # start timer for rk4
for i in range(1,npts):   # loop over time
    y_rwa[i,:] = rk4(derivs_rwa, y_rwa[i-1], tlist[i-1], dt) 

t_rk4 = timeit.default_timer() - ts
print(f"Time for RK4 ODE Solver on pulse:  {round(t_rk4, 4)} s")  

#%% Graphics: Q2

fig = plt.figure(figsize=(8,6))

ax = fig.add_axes([.2, .2, .6, .6])
# plot excitation population for RWA OBE with Gaussian pulse
plt.plot(tlist, y_rwa[:,2], 'r-', label='RK4')#, linewidth=2)
ax.set_xlabel(r'Normalized time, $t/t_p$')
ax.set_ylabel(r'$n_e$')
ax.legend()
# plt.savefig(f'q2.pdf', dpi=1200, bbox_inches="tight")


#%% Question 3
print('\n--- Question 3 ---')

Omega_0 = 2 * np.sqrt(np.pi)  # 2\pi pulse
rabi_f = lambda t: Omega_0 * np.exp(-1 * (t - 5.)**2)  # gaussian pulse with normized time

gamma_d = 0.0
delta_0L = 0.0

# plot n_e(t) curve for a 3 different values of detuning and dephasing
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_axes([.2, .6, .6, .3])  # detuning axis
fmt = ['b-', 'r--', 'g-.']  # line formats
c=0  # counter

max_ne_detuning = np.zeros(100)  # array to populate with max population values
delta_vals = np.linspace(0.0, Omega_0, num=len(max_ne_detuning))  # detuning values to iterate over
print('Varying detuning...')
for i, delta in enumerate(delta_vals):
    delta_0L = delta

    # Run OBE simulation
    dt = 0.01
    tmax = 10.

    tlist = np.arange(0.0, tmax, dt)  # gauarantees the same step size
    npts = len(tlist)
    yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
    y = np.zeros((npts, 3))
    y[0,:] = yinit

    max_ne = -1  # temp value to track maximum population
    for j in range(1,npts):   # loop over time
        y[j,:] = rk4(derivs_rwa, y[j-1], tlist[j-1], dt)

        if y[j, 2] > max_ne:
            max_ne = y[j, 2]

    max_ne_detuning[i] = max_ne
    
    if i in [0, 49, 99]:
        plt.plot(tlist, y[:,2], fmt[c], label=r'{}$\Omega_0$'.format(round(delta/Omega_0, 1)))
        c += 1

# Reset values and run simulation for varied dephasing
gamma_d = 0.0
delta_0L = 0.0

ax2 = fig.add_axes([.2, .2, .6, .3])  # plot dephasing
c=0

max_ne_dephase = np.zeros(100)  # array to populate with max population values
gamma_vals = np.linspace(0.0, Omega_0, num=len(max_ne_dephase))  # dephasing values to iterate over
print('Varying dephasing...')
for i, gamma in enumerate(gamma_vals):
    gamma_d = gamma

    # Run OBE simulation
    dt = 0.01
    tmax = 10.

    tlist = np.arange(0.0, tmax, dt)  # gauarantees the same step size
    npts = len(tlist)
    yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
    y = np.zeros((npts, 3))
    y[0,:] = yinit

    max_ne = -10
    for j in range(1,npts):   # loop over time
        y[j,:] = rk4(derivs_rwa, y[j-1], tlist[j-1], dt)

        if y[j, 2] > max_ne:
            max_ne = y[j, 2]

    max_ne_dephase[i] = max_ne

    if i in [0, 49, 99]:
        plt.plot(tlist, y[:,2], fmt[c], label=r'{}$\Omega_0$'.format(round(gamma/Omega_0, 1)))
        c += 1

# %% Graphics: Q3

# Format the n_e(t) curves collected plotted during simulation
ax1.legend(loc='upper left')
ax1.set_ylabel(r'$n_e$')
ax2.legend()
ax2.set_xlabel(r'Normalized time, $t/t_p$')
ax2.set_ylabel(r'$n_e$')
# plt.savefig(f'q3-full-n.pdf', dpi=1200, bbox_inches="tight")

fig = plt.figure(figsize=(8,6))

ax1 = fig.add_axes([.2, .6, .6, .3])
# Plot effect of detuning on maximum detuning value
plt.plot(delta_vals/Omega_0, max_ne_detuning, 'r-')
ax1.set_xlabel(r'Detuning, $\Delta_{0L} / \Omega_0$')
ax1.set_ylabel(r'max($n_e$)')

ax2 = fig.add_axes([.2, .15, .6, .3])
# Plot effect of dephasing on maximum detuning value
plt.plot(gamma_vals/Omega_0, max_ne_dephase, 'g-')
ax2.set_xlabel(r'Dephasing, $\gamma_d / \Omega_0$')
ax2.set_ylabel(r'max($n_e$)')
# plt.savefig(f'q3.pdf', dpi=1200, bbox_inches="tight")


#%% Question 4 (a)

print('\n--- Question 4 ---')

delta_0L = 0  # laser frequency
gamma_d = 0   # dephasing or polarization dephasing
phi = 0       # driver wave phase
Omega_0 = 2 * np.sqrt(np.pi)  # 2\pi pulse area
coeffs = np.array([20, 10, 5, 2])
omega_L_vals = Omega_0 * coeffs  # list of laser frequencies to test

# Plot different omega_L with Re[u], Im[u], and n_e subplots
figs = [plt.figure(figsize=(8,6)) for _ in range(len(coeffs))]
axes_re = [fig.add_axes([.2, .7, .6, .2]) for fig in figs]
axes_im = [fig.add_axes([.2, .4, .6, .2]) for fig in figs]
axes_n  = [fig.add_axes([.2, .1, .6, .2]) for fig in figs]

for i, omega in enumerate(omega_L_vals):
    # set frequencies - note that we are on resonance
    omega_L = omega
    omega_0 = omega

    # Run OBE simulation
    dt = 0.01
    tmax = 10

    tlist = np.arange(0.0, tmax, dt)
    npts = len(tlist)
    yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
    y = np.zeros((npts, 3))
    y[0,:] = yinit

    max_ne = -1  # track maximum population
    for j in range(1,npts):   # loop over time
        y[j,:] = rk4(derivs, y[j-1], tlist[j-1], dt)
        
        if y[j,2] > max_ne:
            max_ne = y[j, 2]
    
    print(f'Max ne with {coeffs[i]}*Omega0 is: {max_ne}')
    
    # Plot results for each component
    plt.sca(axes_re[i])
    plt.plot(tlist, y[:, 0], 'b-', label=r'$\phi=0$'.format(int(coeffs[i])))
    plt.sca(axes_im[i])
    plt.plot(tlist, y[:, 1], 'b-', label=r'$\phi=0$'.format(int(coeffs[i])))
    plt.sca(axes_n[i])
    plt.plot(tlist, y[:, 2], 'b-', label=r'$\phi=0$'.format(int(coeffs[i])))


# Add a phase \phi = \pi/2 to the 2*Omega_0 frequency case and simulate again!
phi = np.pi / 2
omega_L = 2 * Omega_0
omega_0 = omega_L
# Run OBE simulation
dt = 0.01
tmax = 10

tlist = np.arange(0.0, tmax, dt)
npts = len(tlist)
yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
y = np.zeros((npts, 3))
y[0,:] = yinit

max_ne = -1
for j in range(1,npts):   # loop over time
    y[j,:] = rk4(derivs, y[j-1], tlist[j-1], dt)
    if y[j,2] > max_ne:
        max_ne = y[j, 2]

print(f'Max ne with phase is: {max_ne}')

#%% Graphics: Q4 (a)

# Since most the data has been plotted we just need to format the axes and plot the phase case
plt.sca(axes_re[-1])
plt.plot(tlist, y[:, 0], 'r--', label=r'$\phi=\frac{\pi}{2}$')
plt.legend(loc='upper left')

plt.sca(axes_im[-1])
plt.plot(tlist, y[:, 1], 'r--', label=r'$\phi=\frac{\pi}{2}$')
plt.legend(loc='upper left')

plt.sca(axes_n[-1])
plt.plot(tlist, y[:, 2], 'r--', label=r'$\phi=\frac{\pi}{2}$')
plt.legend(loc='upper left')

[ax.set_title(r'$\omega_L={}\Omega_0$'.format(int(coeffs[i]))) for i, ax in enumerate(axes_re)]
[ax.set_xlabel(r'Normalized time, $t/t_p$') for ax in axes_n]
[ax.set_ylabel(r'Re$[u]$') for ax in axes_re]
[ax.set_ylabel(r'Im$[u]$') for ax in axes_im]
[ax.set_ylabel(r'$n_e$') for ax in axes_n]

# for i, c in enumerate(coeffs):
#     plt.sca(axes_n[i])
#     plt.savefig(f'q4-a-{c}.pdf', dpi=1200, bbox_inches="tight")


#%% Question 4 (b)

omega_L = 4 * np.sqrt(np.pi)  # 4\pi pulse area
omega_0 = omega_L             # on resonance
coeffs = np.array([3, 4, 5])
Omega_0_vals = np.sqrt(np.pi) * coeffs  # values of amplitude to vary pulse area
phi = 0  # no phase

# Create figures for Re[u], Im[u], and n_e with different pulse area subplots
figs = [plt.figure(figsize=(8,6)) for _ in range(3)]
axes = [[fig.add_axes([.2, ymin, .6, .2]) for fig in figs]
         for ymin in [.7, .4, .1]]
fmt = ['b', 'g', 'r']  # line formats for each pulse area

for i, omega in enumerate(Omega_0_vals):
    Omega_0 = omega
    
    # Run OBE simulation
    dt = 0.01
    tmax = 10

    tlist = np.arange(0.0, tmax, dt)
    npts = len(tlist)
    yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
    y = np.zeros((npts, 3))
    y[0,:] = yinit

    for j in range(1,npts):   # loop over time
        y[j,:] = rk4(derivs, y[j-1], tlist[j-1], dt)

    for j in range(3):
        plt.sca(axes[i][j])
        plt.plot(tlist, y[:, j], fmt[i], label=r'{}$\pi$'.format(int(coeffs[i])))

#%% Graphics: Q4 (b)

# Format figures
[ax.legend(loc='upper left') for ax_list in axes for ax in ax_list]
labels=[r'Real component of coherence, Re$[u]$', r'Imaginary component of coherence, Im$[u]$', r'Excitation population density, $n_e$']
for i, ax in enumerate(axes[1]):
    plt.sca(ax)
    ax.set_ylabel(labels[i])
    # plt.savefig(f'q4-b-{i}.pdf', dpi=1200, bbox_inches="tight")


#%% Question 4 (c)

# Plot each pulse area result on its own subplot
fig = plt.figure(figsize=(8,6))
axes = [fig.add_axes([.2, ymin, .6, .2]) for ymin in [.7, .4, .1]]
fmt = ['b', 'g', 'r']  # linestyle formats

# Set OBE constants
omega_L = 4 * np.sqrt(np.pi)
omega_0 = omega_L
phi = 0
gamma_d = 0.4
coeffs = np.array([2, 10, 20])
Omega_0_vals = np.sqrt(np.pi) * coeffs

for i, omega in enumerate(Omega_0_vals):
    Omega_0 = omega
    
    # Run OBE simulation
    dt = 0.01
    tmax = 50
    tlist = np.arange(0.0, tmax, dt)
    npts = len(tlist)
    yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)
    y = np.zeros((npts, 3))
    y[0,:] = yinit

    for j in range(1,npts):   # loop over time
        y[j,:] = rk4(derivs, y[j-1], tlist[j-1], dt)

    # Perform fast Fourier transform on Re[u] to analyze power spectrum of propogated pulse
    power_spec_f = np.abs(np.fft.rfft(y[:, 0]))
    freq = np.fft.rfftfreq(npts, d=dt) * 2 * np.pi / omega_L

    plt.sca(axes[i])
    plt.plot(freq, power_spec_f, fmt[i], label=r'{}$\pi$'.format(int(coeffs[i])))

#%% Graphics: Q4 (c)

[ax.set_xlim(-0.1, 6.1) for ax in axes]
[ax.set_ylim(1e-4, 1e2) for ax in axes]
[ax.set_yscale('log') for ax in axes]   # logarithmic scale on the vertical
[ax.legend(loc='lower left') for ax in axes]
axes[-1].set_xlabel(r'Normalized frequency, $\omega/\omega_L$')
axes[1].set_ylabel(r'Polarization power spectrum, $|P(\omega)|$')
# plt.savefig(f'q4-c.pdf', dpi=1200, bbox_inches="tight")


#%% Question 4 (d)

# Set OBE constants
omega_L = 4 * np.sqrt(np.pi)
omega_0 = omega_L
phi = 0.
gamma_d = 0.4
coeffs = np.array([2, 10, 20])
Omega_0_vals = np.sqrt(np.pi) * coeffs

# Run some simulations to compare my RK4 with the scipy ODE solver

dt = 0.01
tmax = 50
for i, omega in enumerate(Omega_0_vals):
    print(f'\nOmega_0 = {coeffs[i]}*\sqrt{{pi}}')
    Omega_0 = omega
    
    # Run OBE simulation
    tlist = np.arange(0.0, tmax, dt)
    npts = len(tlist)
    yinit = np.array([0.0, 0.0, 0.0])  # initial conditions (TLS in ground state)

    ts = timeit.default_timer()  # start scipy timer
    y_scipy = odeint(derivs, yinit , tlist)
    t_scipy = timeit.default_timer() - ts

    ts = timeit.default_timer()  # start scipy timer
    y_my = np.zeros((npts, 3))
    y_my[0,:] = yinit
    for j in range(1,npts):   # loop over time
        y_my[j,:] = rk4(derivs, y_my[j-1], tlist[j-1], dt)
    t_my = timeit.default_timer() - ts

    abs_diff_u = np.sum(np.abs(y_scipy[:, 0] - y_my[:, 0])) / len(tlist)
    abs_diff_n = np.sum(np.abs(y_scipy[:, 2] - y_my[:, 2])) / len(tlist)

    print(f'My code time: {round(t_my, 4)} s')
    print(f'Scipy time:   {round(t_scipy, 4)} s')
    print(f'My solver took {round(t_my - t_scipy, 4)} seconds longer')
    print(f'My solver took {round(t_my / t_scipy, 2)} times as long')
    print(f'With an absolute error of {round(abs_diff_u, 4)} on Re[u]')
    print(f'With an absolute error of {round(abs_diff_n, 4)} on populaiton')

