# -*- coding: utf-8 -*-
"""
Simple Harmonic Oscillators and Planar Three-Body Dynamics

This file has my code for problem set 1 of ENPH 479.

Created on Sun Jan 30 15:47:24 2022

@author: Matt Wright
"""


from tkinter.tix import TList
import numpy as np 
import timeit 
from scipy.integrate import odeint
import matplotlib.pyplot as plt  
import matplotlib.animation as animation
from matplotlib import rcParams

plt.rcParams.update({'font.size': 20})
# %matplotlib auto


# niceFigure example with good reasonable size fonts and TeX fonts
def niceFigure(useLatex=True):
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return 


def rk4(diff_eq, y, t, h):
    """Vectorized 4th order Runge-Kutta ODE solver. Returns the approximate state of the
    system a time step later.
    
    Parameters
    ----------
    diff_eq : callable
        Function of y (vector) and t (scalar) that returns the coupled partial dertivatives
        of the system.
    y : numpy.ndarray
        Current state of the target system. Should be a 1D array.
    t : float
        Current time of the system.
    h : float
        Time step used by ODE solver. Recommended 100 to 1000 steps per period.
    """

    k1 = h * diff_eq(y, t)
    k2 = h * diff_eq(y + k1/2, t + h/2)
    k3 = h * diff_eq(y + k2/2, t + h/2)
    k4 = h * diff_eq(y + k3, t + h)
    
    y += (k1 + 2.*k2 + 2.*k3 + k4) / 6.
    return y

def sho_derivs(y, t):
    """Coupled ODEs of a Simple Harmonic Oscillator (m=k=omega_0=0). Assumes y is 1D of
    position and velocity."""

    dy = np.zeros(len(y))
    dy[0] = y[1]  # dx/dt = v(t)
    dy[1] = -y[0]   # dv/dt = -x(t)
    return dy


# TODO: comment
def leapfrog(diff_eq , r_0 , v_0 , t, h):
    """Vectorized leapfrog method using numpy arrays.
    
    Parameters
    ----------

    """

    r_12 = r_0 + h/2 * diff_eq(0, r_0, v_0, t)    # r_{1/2} at h/2 using v0=dx/dt
    v_1 = v_0 + h * diff_eq(1, r_12, v_0, t+h/2)  # v_1 using a(r) at h/2
    r_1 = r_12 + h/2 * diff_eq(0, r_0, v_1, t+h)  # r_1 at h using v_1

    return r_1, v_1

def oscillator(id, x, v, t):
    """Simply Harmonic Oscillator equations of motion (m=k=omega_0=0). For id==0, return
    first derivative of x, otherwise return the second derivative."""

    if id == 0:
        return v
    else:
        return -x


def sho_energy(x, v):
    return (x**2 + v**2) / 2



sim_time = 40
h = 0.1


# def run_sho_sim(sim_time, h):
# Set graphics
niceFigure(False)
tpause = 0.1
fig = plt.figure(figsize=(10,8))
plt.ion()
# RK4 phase space solution
ax1 = fig.add_axes([.2, .6, .3, .3])
ax1.set_ylim(-1.1, 1.1)
ax1.set_xlim(-1.1, 1.1)
ax1.set_title('RK4')
ax1.set_ylabel('$v$ (m/s)')
# RK4 Energy
ax2 = fig.add_axes([.6, .6, .3, .3])
ax2.set_ylim(-0.1, 0.7)
# ax2.set_xlim(-1.1, 1.1)
ax2.set_title('RK4')
ax2.set_ylabel('$E$ (J)')
# Leapfrog phase space solution
ax3 = fig.add_axes([.2, .2, .3, .3])
ax3.set_ylim(-1.1, 1.1)
ax3.set_xlim(-1.1, 1.1)
ax3.set_title('Leapfrog')
ax3.set_ylabel('$v$ (m/s)')
ax3.set_xlabel('$x$ (m)')
# Leapfrog Energy
ax4 = fig.add_axes([.6, .2, .3, .3])
ax4.set_ylim(-0.1, 0.7)
# ax2.set_xlim(-1.1, 1.1)
ax4.set_title('Leapfrog')
ax4.set_ylabel('$E$ (J)')
ax4.set_xlabel('$t$ (s)')

T_0 = 2 * np.pi  # period
t_max = sim_time * T_0  # maximum simulation time
t = 0.0  # time tracker
ic = 0   # increments each iteration for plotting frequency
update_f = 10  # number of repetitions before updating plot
nframes = int(t_max / h)
print(f'nframes: {nframes}')
# Specify and set inital conditions
x_0 = 1.0
v_0 = 0.0
x_rk4, v_rk4 = x_0, v_0
x_leapfrog, v_leapfrog = x_0, v_0

E_rk4 = sho_energy(x_rk4, v_rk4)
E_leapfrog = sho_energy(x_leapfrog, v_leapfrog)

rk4_traj, = ax1.plot([], [])
rk4_Et, = ax2.plot([], [])
leapfrog_traj, = ax3.plot([], [])
leapfrog_Et, = ax4.plot([], [])

def init():
    rk4_traj.set_data(x_rk4, v_rk4)
    rk4_Et.set_data(t, E_rk4)
    
    leapfrog_traj.set_data(x_rk4, v_rk4)
    leapfrog_Et.set_data(t, E_leapfrog)
    return rk4_traj, rk4_Et, leapfrog_traj, leapfrog_Et

x_rk4, v_rk4 = [x_0], [v_0]
x_leapfrog, v_leapfrog = [x_0], [v_0]
# E_rk4 = 
def animate(i):
    # global x_rk4, v_rk4, x_leapfrog, v_leapfrog, E_rk4, E_leapfrog, t
    global t

    y_rk4 = np.array([x_rk4[-1], v_rk4[-1]])  # temp array to hold previous vals
    x_rk4_tmp, v_rk4_tmp = rk4(sho_derivs, y_rk4, t, h)
    E_rk4 = sho_energy(x_rk4_tmp, v_rk4_tmp)
    x_rk4.append(x_rk4_tmp)
    v_rk4.append(v_rk4_tmp)

    # Evolve with leapfrog
    x_leapfrog_tmp, v_leapfrog_tmp = leapfrog(oscillator, x_leapfrog[-1], v_leapfrog[-1], t, h)
    E_leapfrog = sho_energy(x_leapfrog_tmp, v_leapfrog_tmp)
    x_leapfrog.append(x_leapfrog_tmp)
    v_leapfrog.append(v_leapfrog_tmp)

    rk4_traj.set_data(x_rk4, v_rk4)
    # rk4_Et.set_data(t, E_rk4)
    
    leapfrog_traj.set_data(x_rk4, v_rk4)
    # leapfrog_Et.set_data(t, E_leapfrog)

    t += 1
    return rk4_traj, rk4_Et, leapfrog_traj, leapfrog_Et

ani = animation.FuncAnimation(fig, animate, frames=nframes, init_func=init, blit=True)
plt.show()


# plt.pause(tpause)
'''
while t < t_max:
    # Evolve with rk4
    y_rk4 = np.array([x_rk4, v_rk4])  # temp array to hold previous vals
    x_rk4, v_rk4 = rk4(sho_derivs, y_rk4, t, h)
    E_rk4 = sho_energy(x_rk4, v_rk4)

    # Evolve with leapfrog
    x_leapfrog, v_leapfrog = leapfrog(oscillator, x_leapfrog, v_leapfrog, t, h)
    E_leapfrog = sho_energy(x_leapfrog, v_leapfrog)
    
    if ic % update_f == 0:
        rk4_traj.set_xdata(x_rk4)
        rk4_traj.set_ydata(v_rk4)
        rk4_Et.set_xdata(t)
        rk4_Et.set_ydata(E_rk4)
        
        leapfrog_traj.set_xdata(x_rk4)
        leapfrog_traj.set_ydata(v_rk4)
        leapfrog_Et.set_xdata(t)
        leapfrog_Et.set_ydata(E_leapfrog)

        plt.draw()
        plt.pause(tpause) # pause to see animation as code v. fast
    
    t += h   # increment time
    ic += 1  # simple integer counter that migth be useful 
'''
# sim_time = 40
# h = 0.1
# run_sho_sim(sim_time, h)