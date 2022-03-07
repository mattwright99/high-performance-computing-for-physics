# -*- coding: utf-8 -*-
"""
Simple Harmonic Oscillators and Planar Three-Body Dynamics

This file has my code for problem set 1 of ENPH 479.

Created on Sun Jan 30 15:47:24 2022

@author: Matt Wright
"""


from inspect import istraceback
from mimetypes import init
from tkinter.tix import TList
from turtle import color
import numpy as np 
import timeit 
from scipy.integrate import odeint
from scipy.optimize import newton
import matplotlib.pyplot as plt  
import matplotlib.animation as animation
from matplotlib import rcParams
import pdb

plt.rcParams.update({'font.size': 20})


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


def sho_derivs(y, t):
    """Coupled ODEs of a Simple Harmonic Oscillator (m=k=omega_0=0). Assumes y is 1D of
    position and velocity."""

    dy = np.zeros(len(y))
    dy[0] = y[1]  # dx/dt = v(t)
    dy[1] = -y[0]   # dv/dt = -x(t)
    return dy

def oscillator(id, x, v, t):
    """Simply Harmonic Oscillator equations of motion (m=k=omega_0=0). For id==0, return
    first derivative of x, otherwise return the second derivative."""

    if id == 0:  # first eqn => dx/dt
        return v
    # Second ode => dv/dt
    return -x

def sho_energy(x, v):
    return (x**2 + v**2) / 2

def run_sho_sim(sim_time, h, show_live=True):
    # Set graphics
    niceFigure(False)
    tpause = 0.1
    if show_live:  # flag to show live simulation
        fig = plt.figure(figsize=(10,8))
        # plt.ion()
        # Dynamic RK4 phase space current state
        ax1 = fig.add_axes([.2, .6, .3, .3])
        ax1.set_ylim(-1.1, 1.1)
        ax1.set_xlim(-1.1, 1.1)
        ax1.set_title('RK4')
        ax1.set_ylabel('$v$ (m/s)')
        # Dynamic RK4 totale phase space tracetory
        ax2 = fig.add_axes([.6, .6, .3, .3])
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_xlim(-1.1, 1.1)
        ax2.set_title('RK4')
        ax2.set_ylabel('$v$ (m/s)')
        # Dynamic Leapfrog phase space current state
        ax3 = fig.add_axes([.2, .2, .3, .3])
        ax3.set_ylim(-1.1, 1.1)
        ax3.set_xlim(-1.1, 1.1)
        ax3.set_title('Leapfrog')
        ax3.set_ylabel('$v$ (m/s)')
        ax3.set_xlabel('$x$ (m)')
        # Dynamic Leapfrog totale phase space tracetory
        ax4 = fig.add_axes([.6, .2, .3, .3])
        ax4.set_ylim(-1.1, 1.1)
        ax4.set_xlim(-1.1, 1.1)
        ax4.set_title('Leapfrog')
        ax4.set_ylabel('$v$ (m/s)')
        ax4.set_xlabel('$x$ (m)')

    T_0 = 2 * np.pi  # period
    h = h * T_0
    t_max = sim_time * T_0  # maximum simulation time
    t = 0.0  # time tracker
    t_list = [t]
    ic = 0   # increments each iteration for plotting frequency
    frames_per_T = 10  # desired number of frames per period
    update_f = np.ceil(T_0 / h / frames_per_T)  # number of repetitions before updating plot

    
    # Specify and set inital conditions
    x_0 = 1.0
    v_0 = 0.0
    x_rk4, v_rk4 = [x_0], [v_0]
    x_leapfrog, v_leapfrog = [x_0], [v_0]

    if show_live:
        # Create plot objects for each axis
        rk4_curr, = ax1.plot(x_rk4[-1], v_rk4[-1], 'rx', markersize=13)
        rk4_traj, = ax2.plot(x_rk4, v_rk4,'b-', markersize=13)
        leapfrog_curr, = ax3.plot(x_leapfrog[-1], v_leapfrog[-1], 'rx', markersize=13)
        leapfrog_traj, = ax4.plot(x_leapfrog, v_leapfrog, 'b-', markersize=13)

    plt.pause(2)
    while t < t_max:
        # Evolve with rk4
        x = x_rk4[-1]  # current x
        v = v_rk4[-1]  # current v
        y = np.array([x, v])  # temp array to hold previous vals for RK4
        x, v = rk4(sho_derivs, y, t, h)
        # append new state data
        x_rk4.append(x)
        v_rk4.append(v)

        # Evolve with leapfrog
        x = x_leapfrog[-1]  # current x
        v = v_leapfrog[-1]  # current v
        x, v = leapfrog(oscillator, x, v, t, h)
        # append new state data
        x_leapfrog.append(x)
        v_leapfrog.append(v)
        
        if ic % update_f == 0 and show_live:
            # Update plots with new data
            rk4_curr.set_data(x_rk4[-1], v_rk4[-1])
            rk4_traj.set_data(x_rk4, v_rk4)
            leapfrog_curr.set_data(x_leapfrog[-1], v_leapfrog[-1])
            leapfrog_traj.set_data(x_leapfrog, v_leapfrog)

            plt.draw()
            plt.pause(tpause) # pause to see animation as code v. fast
        
        t += h
        t_list.append(t)   # increment time
        ic += 1  # simple integer counter that migth be useful 

    # Compute energy throughout simulation
    x_rk4 = np.array(x_rk4)
    v_rk4 = np.array(v_rk4)
    E_rk4 = sho_energy(x_rk4, v_rk4)
    x_leapfrog = np.array(x_leapfrog)
    v_leapfrog = np.array(v_leapfrog)
    E_leapfrog = sho_energy(x_leapfrog, v_leapfrog)

    t_list  = [t/T_0 for t in t_list]  # convert to units of T_0

    # Plot summary energy and total trajectory information
    fig = plt.figure(figsize=(10,8))
    # RK4 phase space solution
    ax1 = fig.add_axes([.2, .6, .3, .3])
    ax1.plot(x_rk4, v_rk4, 'b-')
    # ax1.set_ylim(-1.2, 1.2)
    # ax1.set_xlim(-1.2, 1.2)
    ax1.set_title('RK4')
    ax1.set_ylabel('$v$ (m/s)')
    # RK4 Energy
    ax2 = fig.add_axes([.6, .6, .3, .3])
    ax2.plot(t_list, E_rk4)
    ax2.set_ylim(-0.1, E_rk4.max() + 0.1)
    # ax2.set_xlim(-1.1, 1.1)
    ax2.set_title('RK4')
    ax2.set_ylabel('$E$ (J)')
    # Leapfrog phase space solution
    ax3 = fig.add_axes([.2, .2, .3, .3])
    ax3.plot(x_leapfrog, v_leapfrog, 'b-')
    # ax1.set_ylim(-1.2, 1.2)
    # ax1.set_xlim(-1.2, 1.2)
    ax3.set_title('Leapfrog')
    ax3.set_ylabel('$v$ (m/s)')
    ax3.set_xlabel('$x$ (m)')
    # Leapfrog Energy
    ax4 = fig.add_axes([.6, .2, .3, .3])
    ax4.plot(t_list, E_leapfrog)
    ax4.set_ylim(-0.1, E_leapfrog.max() + 0.1)
    # ax2.set_xlim(-1.1, 1.1)
    ax4.set_title('Leapfrog')
    ax4.set_ylabel('$E$ (J)')
    ax4.set_xlabel('$t$ (s)')
    plt.show()


sim_time = 40
h = 0.2
# run_sho_sim(sim_time, h, show_live=False)
h = 0.02
# run_sho_sim(sim_time, h, show_live=False)


#%% Question 2


def three_body_rk4_derivs(y, t):
    """Three-body planar dynamics equations of motion. Used for ODE solvers that expect a
    1 dimensional input like my RK4 and scipy's odeint."""

    if y.shape != (12, ):
        print(f'Shape must ne (12,) but got {y.shape}')

    dy = np.zeros(y.shape)
    # Update position
    dy[:6] = y[6:]
    # Update velocity
    for indices in [(0,1,2), (1,2,0), (2,0,1)]:
        i, j, k = indices  # indeces for mass
        iy, jy, ky = [2 * idx for idx in indices]  # transformed indices for y vector
        
        r_ij = y[[iy, iy+1]] - y[[jy, jy+1]]
        r_ik = y[[iy, iy+1]] - y[[ky, ky+1]]
        for component in [0, 1]:
            dy[6 + iy + component] = -G * m[j] * r_ij[component] / np.linalg.norm(r_ij)**3 \
                                    - G * m[k] * r_ik[component] / np.linalg.norm(r_ik)**3
    return dy

def three_body_lf_ode(id, r, v, t):
    """Equations of motion for a three-body planatary system."""

    if id == 0:
        return v
    
    dv_dt = np.zeros(v.shape)
    for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
        r_ij = r[i] - r[j]
        r_ik = r[i] - r[k]
        dv_dt[i] = -G * m[j] * r_ij / np.linalg.norm(r_ij)**3 \
                  - G * m[k] * r_ik / np.linalg.norm(r_ik)**3
    return dv_dt

def three_body_energy(y):
    """Compute energy (in units of G=1) of three-body system given a 1D input."""

    # Seperate into position and velocity components
    r = y[:6].reshape((n_bodies, dim))
    v = y[6:].reshape((n_bodies, dim))

    # Compute pair wise gravitational potential energy
    PE = 0.0
    for i in range(len(r)):
        for j in range(i):
            PE += m[i] * m[j] / np.linalg.norm(r[i] - r[j])
    
    Energy = 0.5 * np.sum(m * v.T**2) - PE
    return Energy

def update_bounds(min_max, vals):
    """Check for new extrema for the plot axes limits. Works for both E and r/v"""

    if 1.1 * vals.min() < min_max[0]:
        min_max[0] = 1.1 * vals.min()
    if 1.1 * vals.max() > min_max[1]:
        min_max[1] = 1.1 * vals.max()
    return min_max

def solve_quintic():
    # Solve for roots of Euler's quintic equation
    quintic_eqn = lambda x: x**5 * (m[1]+m[2]) + x**4 * (2*m[1]+3*m[2]) \
                          + x**3 * (m[1]+3*m[2]) - x**2 * (3*m[0]+m[1]) \
                          - x * (3*m[0]+2*m[1]) - (m[0] + m[1])
    # first derivative
    quintic_eqn_deriv = lambda x: 5 * x**4 * (m[1]+m[2]) + 4 * x**3 * (2*m[1]+3*m[2]) \
                                + 3 * x**2 * (m[1]+3*m[2]) - 2 * x**1 * (3*m[0]+m[1]) \
                                - (3*m[0]+2*m[1])
    # second derivative
    quintic_eqn_deriv2 = lambda x: 20 * x**3 * (m[1]+m[2]) + 12 * x**2 * (2*m[1]+3*m[2]) \
                                  + 6 * x**1 * (m[1]+3*m[2]) - 2 * (3*m[0]+m[1])

    sol = newton(quintic_eqn,  x0=0.7, fprime=quintic_eqn_deriv, fprime2=quintic_eqn_deriv2)
    print(f'Quintic Equation Soln: \u03BB = {sol}')
    return sol

def initial_cond_1(omega):
    lam = solve_quintic()  # root of Euler quintic eqn - lambda
    # Compute a - the distance between masses m2 and m3
    a = ((m[1] + m[2] - m[0] * (1+2*lam) / lam**2 / (1+lam)**2) / omega**2) ** (1/3)
    print(f'Distance a = x3 - x2 = {a}')

    # Initial positions
    x_2 = (m[0] / lam**2 - m[2]) / omega**2 / a**2
    x_1 = x_2 - lam * a
    x_3 = -(m[0] * x_1 + m[1] * x_2) / m[2]
    # Initial velocities
    v_1y = omega * x_1
    v_2y = omega * x_2
    v_3y = omega * x_3

    # Create position and velocity vectors for the system
    r = np.zeros((n_bodies, dim))
    v = np.zeros((n_bodies, dim))
    r[0, 0] = x_1
    r[1, 0] = x_2
    r[2, 0] = x_3
    v[0, 1] = v_1y
    v[1, 1] = v_2y
    v[2, 1] = v_3y
    return r, v

def initial_cond2():
    r , v = np . zeros (( 3 , 2 ) ) , np . zeros (( 3 , 2 ) )
    # initial r and v - set 2
    r[0 , 0] = 0.97000436 ; r[0 , 1] = -0.24308753 # x1 , y1
    v[2 , 0] = -0.93240737 ; v[2 , 1] = -0.86473146 # v3x , v3y
    v[0 , 0] = -v[2 , 0]/2.; v[0 , 1] = -v[2 , 1]/2. # v1x , v1y
    r[1 , 0] = - r[0 , 0] ; r[1 , 1]=-r[0 , 1] # x2 , y2
    v[1 , 0] = v[0 , 0]; v[1 , 1] = v[0 , 1] # v2x , v2y
    return r , v

def rk4_solver(diff_eq, y_init, t_list):
    npts = len(t_list)
    result = np.zeros((npts, len(y_init)))
    result[0, :] = y_init
    # Iterate over each time step and evolve the system using rk4 and the provided equations of motion
    for i in range(1, npts):
        dt = t_list[i] - t_list[i-1]
        result[i, :] = rk4(diff_eq, result[i-1], t_list[i-1], dt)
    
    return result

def leapfrog_solver(diff_eq, r, v, t_list):
    npts = len(t_list)
    result = np.zeros((npts, 2*dim*n_bodies))
    result[0, :] = np.array([r, v]).reshape(( 2*dim*n_bodies,))
    # Iterate over each time step and evolve the system using leapfrog and the provided equations of motion
    for i in range(1, npts):
        dt = t_list[i] - t_list[i-1]
        r, v = leapfrog(diff_eq, r, v, t_list[i-1], dt)
        result[i, :] = np.array([r, v]).reshape(( 2*dim*n_bodies,))

    return result

def animate_three_body_sim(t_list, res, energy, show_live, update_f, plt_title):
    # Set graphics
    niceFigure(False)
    tpause = 0.1
    min_max_x = [-2, 2]
    min_max_y = [-2, 2]
    min_max_e = [(1 + 1e-12) * energy.min(), (1 - 1e-12) * energy.max()]

    # Set figure for simulation visualization
    fig = plt.figure(figsize=(10,8))
    # Dynamic interaction plot
    ax = fig.add_axes([.2, .1, .6, .5])
    ax.set_ylabel('$y$ (arb. units)')
    ax.set_xlabel('$x$ (arb. units)')
    ax.set_ylim(*min_max_y)
    ax.set_xlim(*min_max_x)
    if plt_title:
        ax.title(plt_title)
    # Energy plot
    ax2 = fig.add_axes([.2, .7, .6, .2])
    ax2.set_ylabel('$E$ (arb. units)')
    ax2.set_xlabel('Time ($T_0$)')
    ax2.set_ylim(*min_max_e)


    # Plot initial state and instantiate plot objects
    colors = ['r', 'b', 'g']
    # Plot initial points and CoM
    ax.scatter(0, 0, facecolors='none', edgecolors='k', s=25) 
    [ax.scatter(res[0, 2*i], res[0, 2*i+1], c=colors[i], s=15) for i in range(n_bodies)]
    # Plot objects for total planet trajectories
    trajectories = [ax.plot(res[0, 2*i], res[0, 2*i+1], colors[i]+'--')[0] for i in range(n_bodies)]
    # Quiver objects for each planets position and direction
    Qs = [
        ax.quiver(res[0, 2*i], res[0, 2*i+1], res[0, 2*i+6], res[0, 2*i+7], width=0.005, color=colors[i],  
            headwidth=8, headlength=3, headaxislength=3, scale=40)
        for i in range(n_bodies)
    ]
    # Energy v time plot
    e_plot, = ax2.plot(t_list[0], energy[0], 'r-')

    if show_live:
        plt.pause(2)
        for i in range(1, len(res)):
            if i % update_f == 0:
                [Q.remove() for Q in Qs]  # remove quivers
                [traj.set_data(res[:i, 2*j], res[:i, 2*j+1]) for j, traj in enumerate(trajectories)]  # plot new position data
                # Update quivers
                Qs = [
                    ax.quiver(res[i, 2*j], res[i, 2*j+1], res[i, 2*j+6], res[i, 2*j+7], width=0.005, color=colors[j],  
                        headwidth=8, headlength=3, headaxislength=3, scale=50)
                    for j in range(n_bodies)    
                ]
                min_max_x = update_bounds(min_max_x, res[i-update_f : i, [0, 2, 4]])
                min_max_y = update_bounds(min_max_y, res[i-update_f : i, [1, 3, 5]])
                # update axes
                ax.set_xlim(*min_max_x)
                ax.set_ylim(*min_max_y)

                # plot energy and update axes
                e_plot.set_data(t_list[:i], energy[:i])
                ax2.set_xlim(right=t_list[i] + .2)

                plt.draw()
                plt.pause(tpause) # pause to see animation as code v. fast

    # Plot final total state
    [Q.remove() for Q in Qs]  # remove quivers
    [traj.set_data(res[:, 2*j], res[:, 2*j+1]) for j, traj in enumerate(trajectories)]  # plot new position data
    # Update quivers
    Qs = [
        ax.quiver(res[i, 2*j], res[:, 2*j+1], res[:, 2*j+6], res[:, 2*j+7], width=0.005, color=colors[j],  
            headwidth=8, headlength=3, headaxislength=3, scale=50)
        for j in range(n_bodies)    
    ]
    min_max_x = update_bounds(min_max_x, res[:, [0, 2, 4]])
    min_max_y = update_bounds(min_max_y, res[:, [1, 3, 5]])
    # update axes
    ax.set_xlim(*min_max_x)
    ax.set_ylim(*min_max_y)

    # plot energy and update axes
    e_plot.set_data(t_list[:i], energy[:i])
    ax2.set_xlim(right=t_list[i] + .2)
    plt.show()

def run_three_body_sim(
        sim_time, 
        h,
        omega=1,
        ode_solver='rk4', 
        show_live=True,
        frames_per_T=10, 
        init_cond=1, 
        plt_title=None
    ):
    """Run a planar three-body dynamics simulation.

    Parameters
    ----------
    sim_time : int
        Time, in units of period (T=2\pi/omega_0), to run the simulation.
    h : float
        Time step (in units of period) to use for solving the three-body equations of 
        motion.
    omega : float
        Angular frequency of the system.
    ode_solver : str
        Name of numerical method used to solve the equations of motion. Must be `rk4`,
        `leapfrog`, or `scipy`.
    show_live : bool
        Boolean flag to show live animation (if `True`) or just the final result (if
        `False`).
    frames_per_T : int
        Specifies the number of steps per period to update the live animation.
    init_cond : int
        Specifies the initial conditions on position and velocity used for the simulation.
    plt_title : str
        Title used for the plot.
    """
    
    # Time settings
    update_f = int(np.ceil(1 / (h * frames_per_T)))  # number of steps before updating plot
    T_0 = 2 * np.pi / omega  # period
    h = h * T_0
    t_max = sim_time * T_0  # maximum simulation time
    t = 0.0  # currnet time tracker
    t_list = [t]  # time array
    ic = 0   # increments each iteration for plotting frequency
    
    if init_cond == 1:
        r, v = initial_cond_1(omega)
    elif init_cond == 2:
        r, v = initial_cond2()
    else:
        print(f'Initial condition value of 1 or 2 required. Got {init_cond}')

    # min_max_x = update_bounds(min_max_x, r, 0)
    # min_max_y = update_bounds(min_max_y, r, 1)
    # min_max_e = update_bounds(min_max_e, energy)
    
    # # Create arrays for each r and v component
    # r_total = [[[r0] for r0 in ri] for ri in r]
    # v_total = [[[v0] for v0 in vi] for vi in v]

    # # Plot initial state and instantiate plot objects
    # colors = ['r', 'b', 'g']
    # # Plot initial points and CoM
    # ax.scatter(0, 0, facecolors='none', edgecolors='k', s=25) 
    # [ax.scatter(r[i, 0], r[i, 1], c=colors[i], s=15) for i in range(n_bodies)]
    # # Plot objects for total planet trajectories
    # trajectories = [ax.plot(r[i, 0], r[i, 1], colors[i]+'--')[0] for i in range(n_bodies)]
    # # Quiver objects for each planets position and direction
    # Qs = [
    #     ax.quiver(r[i, 0], r[i, 1], v[i, 0], v[i, 1], width=0.005, color=colors[i],  
    #         headwidth=8, headlength=3, headaxislength=3, scale=40)
    #     for i in range(n_bodies)
    # ]
    # # Energy v time plot
    # e_plot, = ax2.plot(t_list, energy, 'r-')

    if show_live:
        plt.pause(2)

    t_list = np.arange(0.0, t_max, h)
    if ode_solver == 'rk4':
        # Evolve each planets position and velocity with my RK4 ODE solver
        y_init = np.array([r, v]).reshape((2 * dim * n_bodies, ))  # temp array to hold previous vals for RK4
        result = rk4_solver(three_body_rk4_derivs, y_init, t_list)
    elif ode_solver == 'leapfrog':
        result = leapfrog_solver(three_body_lf_ode, r, v, t_list)
    elif ode_solver == 'scipy':
        # Evolve each planets position and velocity with scipy's odeint solver
        y_init = np.array([r, v]).reshape((2 * dim * n_bodies, ))  # temp array to hold previous vals for solver
        result = odeint(three_body_rk4_derivs, y_init, t_list, atol=1e-12, rtol=1e-12)

    t_plot = t_list / T_0  # plot in normalized time
    energy = np.array([three_body_energy(y) for y in result]) 
    animate_three_body_sim(t_plot, result, energy, show_live, update_f, plt_title)
    asdfasd


    while t < t_max:
        if ode_solver == 'rk4':
            # Evolve each planets position and velocity with RK4
            y = np.array([r, v]).reshape(( 2*dim*n_bodies,))  # temp array to hold previous vals for RK4
            y = rk4(three_body_derivs, y, t, h)
            # Extract r and v data from the new 1D array information
            r = y[:6].reshape((n_bodies, dim))
            v = y[6:].reshape((n_bodies, dim))
        elif ode_solver == 'leapfrog':
            # Evolve with leapfrog
            r, v = leapfrog(three_body_lf_ode, r, v, t, h)
        else:
            print(f'Invalid ODE sovler given. Must be "rk4", "leapfrog", or "scipy" but got: "{ode_solver}".')
        
        # Append new state data
        for i in range(n_bodies * dim):
            row = i % n_bodies
            col = i % dim
            r_total[row][col].append(r[row, col])
            v_total[row][col].append(v[row, col])

        energy.append(three_body_energy(r, v))

        # increment time
        t += h
        t_list.append(t / T_0)   

        # Update min max values
        min_max_x = update_bounds(min_max_x, r, 0)
        min_max_y = update_bounds(min_max_y, r, 1)
        min_max_e = update_bounds(min_max_e, energy)

        # Update plot if showing animation
        if show_live and ic % update_f == 0:
            # Update plots with new data
            [Q.remove() for Q in Qs]  # remove quivers
            [traj.set_data(*r_total[i]) for i, traj in enumerate(trajectories)]  # plot new position data
            # Update quivers
            Qs = [
                ax.quiver(r[i, 0], r[i, 1], v[i, 0], v[i, 1], width=0.005, color=colors[i],  
                    headwidth=8, headlength=3, headaxislength=3, scale=50)
                for i in range(n_bodies)    
            ]
            # update axes
            ax.set_xlim(*min_max_x)
            ax.set_ylim(*min_max_y)

            # plot energy and update axes
            e_plot.set_data(t_list, energy)
            ax2.set_xlim(right=t_list[-1] + .2)
            ax2.set_ylim(*min_max_e)

            # if show_live:
            plt.draw()
            plt.pause(tpause) # pause to see animation as code v. fast

        ic += 1  # simple integer counter for the cycle 

    # Show final state of simulation
    [Q.remove() for Q in Qs]  # remove quivers
    [traj.set_data(*r_total[i]) for i, traj in enumerate(trajectories)]  # plot new position data
    # Update quivers
    Qs = [
        ax.quiver(r[i, 0], r[i, 1], v[i, 0], v[i, 1], width=0.005, color=colors[i],  
            headwidth=8, headlength=3, headaxislength=3, scale=50)
        for i in range(n_bodies)    
    ]
    # update axes
    ax.set_xlim(*min_max_x)
    ax.set_ylim(*min_max_y)

    # plot energy and update axes
    e_plot.set_data(t_list, energy)
    ax2.set_xlim(right=t_list[-1] + .2)
    ax2.set_ylim(*min_max_e)
    plt.show()
    return

G = 1
n_bodies = 3  # number of planets
dim = 2  # number of dimensions
m = [1, 2, 3]  # planet mass array: m1->m[0], m2->m[1], etc
# m = [1, 1, 1+1e-6]

sim_time = 6
h = 0.001
omega = 1
delta = 1e-9
solver = 'rk4'

run_three_body_sim(sim_time, h, omega=omega, ode_solver=solver, show_live=True,)
run_three_body_sim(sim_time, h, omega=omega+delta, ode_solver=solver, show_live=True)
# run_three_body_sim(sim_time, h, omega=omega-delta, ode_solver=solver, show_live=False)

# run_three_body_sim(sim_time, h, omega=omega, ode_solver=solver, show_live=False, init_cond=2)


