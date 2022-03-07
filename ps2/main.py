# -*- coding: utf-8 -*-
"""
Simple Harmonic Oscillators and Planar Three-Body Dynamics

This file has my code for Problem Set 2 of ENPH 479.

Created on Sun Jan 30 15:47:24 2022

@author: Matt Wright
"""


import numpy as np 
from time import time
from scipy.integrate import odeint
from scipy.optimize import newton
import matplotlib.pyplot as plt  
from matplotlib import rcParams
import pdb

plt.rcParams.update({'font.size': 15})


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

def leapfrog(diff_eq , r_0 , v_0 , t, h):
    """Vectorized leapfrog method using numpy arrays.
    
    Parameters
    ----------
    diff_eq : callable
        Function of the form f(id, r, v, t) which computes the equations of motion of the
        desired system.
    r_0 : np.ndarray
        Array holding lower derivative (e.g. position) data.
    v_0 : np.ndarray
        Array holding higher derivative (e.g. velocity) data.
    t : float
        Current time of the system.
    h : float
        Time step size used for approximation.
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
    """Compute energy of system: KE + PE"""

    return (x**2 + v**2) / 2

def run_sho_sim(sim_time, h, show_live=False, frames_per_T=10):
    """"Complete a simply harmonic oscillator simulation using leapfrog and RK4 ODE solvers
    
    Parameters
    ----------
    sim_time : float
        Time, in units of period (T=2\pi), to run the simulation.
    h : float
        Time step to use for solving the three-body equations of motion.
    show_live : bool
        Set to True to see the live animation of the system evolution.
    frames_per_T : int
        Desired number of frames per period to use during animation.
    """
    
    # Set graphics
    niceFigure(False)
    tpause = 0.1
    if show_live:  # flag to show live simulation
        fig = plt.figure(figsize=(10,8))
        # plt.ion()
        # Dynamic RK4 phase space current state
        ax1 = fig.add_axes([.2, .6, .3, .3])
        ax1.set_ylim(-1.4, 1.4)
        ax1.set_xlim(-1.4, 1.4)
        ax1.set_title('RK4')
        ax1.set_ylabel('$v$ (m/s)')
        # Dynamic RK4 totale phase space tracetory
        ax2 = fig.add_axes([.6, .6, .3, .3])
        ax2.set_ylim(-1.4, 1.4)
        ax2.set_xlim(-1.4, 1.4)
        ax2.set_title('RK4')
        ax2.set_ylabel('$v$ (m/s)')
        # Dynamic Leapfrog phase space current state
        ax3 = fig.add_axes([.2, .2, .3, .3])
        ax3.set_ylim(-1.4, 1.4)
        ax3.set_xlim(-1.4, 1.4)
        ax3.set_title('Leapfrog')
        ax3.set_ylabel('$v$ (m/s)')
        ax3.set_xlabel('$x$ (m)')
        # Dynamic Leapfrog totale phase space tracetory
        ax4 = fig.add_axes([.6, .2, .3, .3])
        ax4.set_ylim(-1.4, 1.4)
        ax4.set_xlim(-1.4, 1.4)
        ax4.set_title('Leapfrog')
        ax4.set_ylabel('$v$ (m/s)')
        ax4.set_xlabel('$x$ (m)')

    T_0 = 2 * np.pi  # period
    h = h * T_0
    t_max = sim_time * T_0  # maximum simulation time
    t = 0.0  # time tracker
    t_list = [t]
    ic = 0   # increments each iteration for plotting frequency
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
    ax1.set_title('RK4')
    ax1.set_ylabel('$v$ (m/s)')
    # RK4 Energy
    ax2 = fig.add_axes([.6, .6, .3, .3])
    ax2.plot(t_list, E_rk4)
    ax2.set_ylim(-0.1, E_rk4.max() + 0.1)
    ax2.set_title('RK4')
    ax2.set_ylabel('$E$ (J)')
    # Leapfrog phase space solution
    ax3 = fig.add_axes([.2, .2, .3, .3])
    ax3.plot(x_leapfrog, v_leapfrog, 'b-')
    ax3.set_title('Leapfrog')
    ax3.set_ylabel('$v$ (m/s)')
    ax3.set_xlabel('$x$ (m)')
    # Leapfrog Energy
    ax4 = fig.add_axes([.6, .2, .3, .3])
    ax4.plot(t_list, E_leapfrog)
    ax4.set_ylim(-0.1, E_leapfrog.max() + 0.1)
    ax4.set_title('Leapfrog')
    ax4.set_ylabel('$E$ (J)')
    ax4.set_xlabel('$t$ (s)')
    plt.show()


#%% Question 1

sim_time = 40
h = 0.2
run_sho_sim(sim_time, h, show_live=True)
# h = 0.02
run_sho_sim(sim_time, h, show_live=True)


#%% Question 2 - functions and global variables

def three_body_rk4_derivs(y, t):
    """Three-body planar dynamics equations of motion. Used for ODE solvers that expect a
    1 dimensional input like my RK4 and scipy's odeint."""

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
    return dy.copy()

def three_body_lf_ode(id, r, v, t):
    """Equations of motion for a three-body planatary system."""

    if id == 0:
        return v.copy()
    
    dv_dt = np.zeros(v.shape)
    for i, j, k in [(0,1,2), (1,2,0), (2,0,1)]:
        r_ij = r[i] - r[j]
        r_ik = r[i] - r[k]
        dv_dt[i] = -G * m[j] * r_ij / np.linalg.norm(r_ij)**3 \
                  - G * m[k] * r_ik / np.linalg.norm(r_ik)**3
    return dv_dt.copy()

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

def newton_root_solver(f, fprime, x0, error=1e-15, max_iter=50):
    """Find rool of function f using Newton's method."""

    sol = x0
    for i in range(max_iter):
        # Find current value of the function
        f_x = f(sol)
        if abs(f_x) < error:
            return sol
        # Compute first derivative at current x
        f_x_prime = fprime(sol)
        if f_x_prime == 0:
            print('ERROR: Could not find root: Zero derivative')
            return None
        # Update using newtons method
        sol = sol - f_x/f_x_prime
    print('ERROR: Could not find root: Hit max iterations')
    return None

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
    # NOTE: Can use the first and second derivatives to find the root but I get weird results.
    # Also note that changing between the root solvers and thier settings gives widly different results.
    sol = newton(quintic_eqn,  x0=0.0, fprime=quintic_eqn_deriv, fprime2=quintic_eqn_deriv2)
    sol2 = newton_root_solver(quintic_eqn, quintic_eqn_deriv, x0=0.0)
    print(f'Quintic Equation Soln: \u03BB = {sol2}')
    # print(f'Difference between scipy and my root: {sol-sol2}')
    return sol2

def initial_cond_1(omega):
    lam = solve_quintic()  # root of Euler quintic eqn - lambda
    # Compute a - the distance between masses m2 and m3
    a = ((m[1] + m[2] - m[0] * (1+2*lam) / lam**2 / (1+lam)**2) / omega**2) ** (1/3)
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

def rk4_solver(diff_eq, y_init, t_list, reversal=False):
    """Uses the RK4 method to iterate over provided times and evolve the system according
    to the given equations of motion."""
    
    npts = len(t_list)
    result = np.zeros((npts, len(y_init)))
    result[0, :] = y_init
    # Iterate over each time step and evolve the system using rk4 and the provided equations of motion
    for i in range(1, npts):
        dt = t_list[i] - t_list[i-1]
        if reversal and i == (npts // 2):
            # Halfway so reverse velocities
            result[i-1, 6:] =  -1 * result[i-1, 6:]
        result[i, :] = rk4(diff_eq, result[i-1], t_list[i-1], dt)
    
    return result

def leapfrog_solver(diff_eq, r, v, t_list, reversal=False):
    """Uses the Leapfrog method to iterate over provided times and evolve the system
    according to the given equations of motion."""

    npts = len(t_list)
    result = np.zeros((npts, 2*dim*n_bodies))
    result[0, :] = np.array([r, v]).reshape(( 2*dim*n_bodies,))
    # Iterate over each time step and evolve the system using leapfrog and the provided equations of motion
    for i in range(1, npts):
        dt = t_list[i] - t_list[i-1]
        if reversal and i == (npts // 2):
            # Halfway so reverse velocities
            v =  -1 * v
        r, v = leapfrog(diff_eq, r, v, t_list[i-1], dt)
        result[i, :] = np.array([r, v]).reshape((2*dim*n_bodies,))

    return result

def animate_three_body_sim(t_list, res, energy, show_live, update_f, plt_title, snap_shots):
    # Set graphics
    niceFigure(False)
    tpause = 0.1
    num_snapshots = 4  # number of snap shots if snap_shots is True
    # minimum and maximum values for axes
    min_max_x = [-2, 2]
    min_max_y = [-2, 2]
    min_max_e = [0,0]
    if energy.min() < 0:
        min_max_e[0] = (1 + 1e-12) *  energy.min()
    else:
        min_max_e[0] = (1 - 1e-12) *  energy.min()
    if energy.max() < 0:
        min_max_e[1] = (1 - 1e-12) * energy.max()
    else:
        min_max_e[1] = (1 + 1e-12) * energy.max()

    # Set figure for simulation visualization
    fig = plt.figure(figsize=(10,8))
    # Dynamic interaction plot
    ax = fig.add_axes([.2, .1, .6, .5])
    ax.set_ylabel('$y$ (arb. units)')
    ax.set_xlabel('$x$ (arb. units)')
    ax.set_ylim(*min_max_y)
    ax.set_xlim(*min_max_x)
    # Energy plot
    ax2 = fig.add_axes([.2, .7, .6, .2])
    ax2.set_ylabel('$E$ (arb. units)')
    ax2.set_xlabel('Time ($T_0$)')
    ax2.set_ylim(*min_max_e)
    if plt_title:
        ax2.set_title(plt_title, loc='right')

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

    if snap_shots:
        plt.savefig(f'three-body-sim_{time()}.pdf', dpi=1200, bbox_inches="tight")

    if show_live or snap_shots:
        snap_f = len(res) // (num_snapshots-1)  # snap shot frequency (in addition to the initial state)
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

                ax.set_title(f'Frame : {i}')

                plt.draw()
                plt.pause(tpause) # pause to see animation as code v. fast

            if snap_shots and i % snap_f == 0:
                plt.savefig(f'three-body-sim_{time()}.pdf', dpi=1200, bbox_inches="tight")

    # Plot final total state
    [Q.remove() for Q in Qs]  # remove quivers
    [traj.set_data(res[:, 2*j], res[:, 2*j+1]) for j, traj in enumerate(trajectories)]  # plot new position data
    # Update quivers
    Qs = [
        ax.quiver(res[-1, 2*j], res[-1, 2*j+1], res[-1, 2*j+6], res[-1, 2*j+7], width=0.005, color=colors[j],  
            headwidth=8, headlength=3, headaxislength=3, scale=50)
        for j in range(n_bodies)    
    ]
    min_max_x = update_bounds(min_max_x, res[:, [0, 2, 4]])
    min_max_y = update_bounds(min_max_y, res[:, [1, 3, 5]])
    # update axes
    ax.set_xlim(*min_max_x)
    ax.set_ylim(*min_max_y)

    # plot energy and update axes
    e_plot.set_data(t_list, energy)
    ax2.set_xlim(right=t_list[-1] + .2)
    plt.draw()
    plt.show()


def run_three_body_sim(
        sim_time, 
        h,
        omega=1,
        ode_solver='rk4', 
        show_live=True,
        frames_per_T=10, 
        init_cond=1, 
        plt_title=None,
        snap_shots=False,
        reversal=False
    ):
    """Run a planar three-body dynamics simulation.

    Parameters
    ----------
    sim_time : float
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
    snap_shots : bool
        Specifies whether or not to save snapshots of the simulation. If True it saves 4
        snap shots of the simulation.
    reversal : bool
        If True, reverse velocities halfway through simulation.
    """
    
    # Time settings
    T_0 = 2 * np.pi / omega  # period
    # h = h * T_0
    t_max = sim_time * T_0  # maximum simulation time
    t = 0.0  # currnet time tracker
    t_list = [t]  # time array
    update_f = int(np.ceil(T_0 / (h * frames_per_T)))  # number of steps before updating plot
    
    # Set initial conditions
    if init_cond == 1:
        r, v = initial_cond_1(omega)
    elif init_cond == 2:
        r, v = initial_cond2()
    else:
        print(f'Initial condition value of 1 or 2 required. Got {init_cond}')

    # Solve equations of motion
    t_list = np.arange(0.0, t_max+h, h)
    if ode_solver == 'rk4':
        # Evolve each planets position and velocity with my RK4 ODE solver
        y_init = np.array([r, v]).reshape((2 * dim * n_bodies, ))  # temp array to hold intial vals for RK4
        result = rk4_solver(three_body_rk4_derivs, y_init, t_list, reversal)
    elif ode_solver == 'leapfrog':
        # Evolve system with leapfrog ODE solver
        result = leapfrog_solver(three_body_lf_ode, r, v, t_list, reversal)
    elif ode_solver == 'scipy':
        # Evolve position and velocity with scipy's odeint solver
        y_init = np.array([r, v]).reshape((2 * dim * n_bodies, ))  # temp array to hold initial vals for solver
        result = odeint(three_body_rk4_derivs, y_init, t_list, atol=1e-12, rtol=1e-12)
    else:
        print(f'Invalid `ode_solver` provided. Must be `rk4`, `leapfrog`, or `scipy` but got: `{ode_solver}`.')

    t_plot = t_list / T_0  # plot in normalized time
    energy = np.array([three_body_energy(y) for y in result])
    animate_three_body_sim(t_plot, result, energy, show_live, update_f, plt_title, snap_shots)

    return result, energy

G = 1
n_bodies = 3  # number of planets
dim = 2  # number of dimensions (i.e. e and y)
m = [1, 2, 3]  # planet mass array: m1->m[0], m2->m[1], etc

sim_time = 4.
h = 0.001
omega = 1
delta = 1e-9

# NOTE: for each simulation call, change the `show_live` argument to False to not show the
# animation and just observe the total result.
 
#%% Q2 (a) Run the simulation
rk4_res, rk4_e = run_three_body_sim(sim_time, h, omega, ode_solver='rk4', show_live=True, plt_title='RK4 Solver', snap_shots=False)
lf_res, lf_e = run_three_body_sim(sim_time, h, omega, ode_solver='leapfrog', show_live=True, plt_title='Leapfrog Solver')

# Introduce some small deviation to omega
run_three_body_sim(sim_time, h, omega=omega + delta, ode_solver='rk4', show_live=True, plt_title=r'$\omega_0=1+\delta$')
run_three_body_sim(sim_time, h, omega=omega - delta, ode_solver='rk4', show_live=True, plt_title=r'$\omega_0=1-\delta$')


#%% Q2 (b) Try with the scipy `odeint` ODE solver
scipy_res, scipy_e = run_three_body_sim(sim_time, h, omega, ode_solver='scipy', show_live=True, plt_title='Scipy Solver')
rk4_mae = np.sum(np.abs(scipy_res - rk4_res), axis=0) / len(scipy_res)
rk4_e_mae = np.sum(np.abs(scipy_e - rk4_e)) / len(scipy_e)
print(f'RK4 MAE for each quantity is: \n{rk4_mae}')
print(f'RK4 MAE for energy is : {rk4_e_mae}\n')

lf_mae = np.sum(np.abs(scipy_res - lf_res), axis=0) / len(scipy_res)
lf_e_mae = np.sum(np.abs(scipy_e - lf_e)) / len(scipy_e)
print(f'Leapfrog MAE for each quantity is: \n{lf_mae}')
print(f'Leapfrog MAE for energy is : {lf_e_mae}\n')

idx = int(0.75 * len(scipy_res))
lf_mae_end = np.sum(np.abs(scipy_res[idx:] - lf_res[idx:]), axis=0) / len(scipy_res[idx:])
print(f'Leapfrog MAE for last quarter is: \n{lf_mae_end}\n')

print(
""" 
------------------------------------ Q2 (b) Comments ------------------------------------

The Leapfrog method is certainly closer to the results of the scipy ODE solver. This is
visually evident given the difference in departure points between the RK4 solver and the
other two methods. This is also supported by the mean absolute errors of the results of
the simulation. The leapfrog solver has a lower MAE for every position and velocity
component. This is a significant difference (around 3 to 5 fold) for positions but is much
closer for velocities. This is understandable since there is a more limited amount of
kinetic energy in the system but the bodies can depart in many different ways by a much
larger amount. Further, the MAE of the leapfrog result increases significantly in the last
quater of the simulation which is understandable becasue that is the part of the simulation
where the system becomes unstable and any minute change in the system evolution has a
significant impact on the final state.Interestingly, the MAE of the energy for the RK4
solver is much lower than that of leapfrog. I am not sure why this is but suspect that it
is because the leapforg energy oscillates around the proper value while the RK4 energy
remains accuracte but slowly dissipates.

"""
)

#%% Q2 (c) Reverse velocities halfway through
sim_time = 6.5
run_three_body_sim(sim_time, h, omega+delta, ode_solver='rk4', show_live=True, plt_title='RK4 - Time reversal', reversal=True)
res, _ = run_three_body_sim(sim_time, h, omega+delta, ode_solver='leapfrog', show_live=True, plt_title='Leapfrog - Time reversal', reversal=True)
print(f'Initial state of the system: \n{res[0]}')
print(f'Final state of the system: \n{res[-1]}')

print(
""" 
------------------------------------ Q2 (c) Comments ------------------------------------

The reversal of velocity in this simulation is analogous to the time reversal symmetry one
might study in a quantum mechanics course. Since the system lives in a conservative, time-
independent potential then reversing momentum will cause all the particles to retrace
their path. This is exactly what we see in this simulation. Near the halfway mark, the 
bodies begin to deviate and destabilize but then once reversed, each particle retraces its
path. It can also be seen that the energy plot is symmetric about the reversal with some
exception (likely due to numerical error). The RK4 plot begins to destabilze more noticably
than the leapfrog solution but they both return near their initial points. The leapfrog
end result is certainly more stable and closer to the initial state. Looking at the 
printed differences between the initial and final state, it can be seen that the error is
typically on the order of 1e-2. The energy spike seenin the leapfrog solution (at the 
halfway point) is more prominent so the symmetry is more obvious.

"""
)

#%% Q2 (d)
m = [1, 1, 1]
omega = 0.99
sim_time = 4
run_three_body_sim(sim_time, h, omega=omega, ode_solver='leapfrog', show_live=True, init_cond=2, frames_per_T=20)

m = [1, 1, 1 + 1e-6]
sim_time = 8
run_three_body_sim(sim_time, h, omega=omega, ode_solver='leapfrog', show_live=True, init_cond=2, frames_per_T=20)

print(
""" 
------------------------------------ Q2 (d) Comments ------------------------------------

The system is stable even with the longer, 8 period simulation and mass perturbation. The
energy in the leapfrog (my most accurate solver) solution is periodic and so is the RK4
solution. The RK4 energy under these conditions slowly decreases as expected but has many
little bumps that appear periodic (about 6 bumps per period). Similarly, the leapfrog
solution is not a perfect sine function but has small plateau at each trough.

"""
)
