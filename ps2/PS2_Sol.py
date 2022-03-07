#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 12:58:10 2022

@author: kylesinger
"""
import matplotlib.pyplot as plt  
import numpy as np 
import math
from scipy import optimize
# from matplotlib.animation import FuncAnimation

plt.rc('font', size=11, family='serif')
plt.rc('axes', titlesize=11)
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 60 # reset to 600 for nice figures to save
plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.5

# General Function Use
def plot(names, x, y, axes, save=False, filename=''):
    colours = ['r', 'g', 'b', 'y', 'p']
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    plt.figure(figsize=(4, 4))
    for i in range(len(y)):
        plt.plot(x, y[i], colours[i])
        
        if names != []:
            plt.legend(names, loc='lower left', mode="expand", bbox_to_anchor=(0,1.02,1,0.2))
            
        plt.tight_layout()
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
    if save:
        plt.savefig(filename) 
    plt.show() 
    
def subplots(names, x, y, axes, save=False, filename=''):
    colours = ['r', 'g', 'b', 'y', 'p']
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    plt.figure(figsize=(4, 4))
    for i in range(1, len(names)+1):
        plt.figure(figsize=(4, 4))
        plt.subplot(len(names),1,i)
        plt.plot(x,y[i-1], colours[i-1])
        plt.tight_layout()
        plt.xlabel(axes[0])
        plt.ylabel(axes[1])
        plt.legend([names[i-1]], bbox_to_anchor=(0,0), loc="center left")
    if save:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show() 
    
def subplotsax(names, x, y, axes, save=False, filename=''):
    colours = ['r', 'g', 'b', 'y', 'p']
    labels = ['(a)','(b)','(c)','(d)','(e)']
    fig, axs = plt.subplots(nrows=len(y), ncols=1,sharex=True, sharey=True)
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    
    fig.set_figheight(4)
    fig.set_figwidth(4)
    
    for i in range(len(names)): 
        val = r"$\rm "+labels[i]+"$"
        axs[i].plot(x, y[i], colours[i], label=names[i])
        axs[i].set_xlabel(axes[0])
        axs[i].set_ylabel(axes[1]) 
        axs[i].text(-0.22, 1.03, val, transform=axs[i].transAxes, size=12, ha='center')
      
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(names, loc='lower left', mode="expand", ncol=len(names), bbox_to_anchor=(0,1.02,1,0.2))
    fig.tight_layout()
    
    
    if save:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show() 
    
def subplotsax_multi_x(names, x, y, axes, save=False, filename=''):
    colours = ['r', 'g', 'b', 'y', 'p']
    labels = ['(a)','(b)','(c)','(d)','(e)']
    fig, axs = plt.subplots(nrows=len(y), ncols=1)
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    axes[2] = '$'+str(axes[2])+'$'
    axes[3] = '$'+str(axes[3])+'$'
    
    fig.set_figheight(4)
    fig.set_figwidth(4)
    
    for i in range(len(names)): 
        val = r"$\rm "+labels[i]+"$"
        axs[i].plot(x[i], y[i], colours[i], label=names[i])
        axs[i].set_xlabel(axes[2*i])
        axs[i].set_ylabel(axes[2*i+1]) 
        axs[i].text(-0.22, 1.03, val, transform=axs[i].transAxes, size=12, ha='center')
      
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(names, loc='lower left', mode="expand", ncol=len(names), bbox_to_anchor=(0,1.02,1,0.2))
    fig.tight_layout()

    plt.ylim(0, 1) # only for specific plots (q1)
    
    if save:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show() 
    
    
def subplot_grid(names, x, y, axes, save=False, filename=''):
    colours = ['r', 'g', 'b', 'y', 'p', 'm', 'c','k']
    labels = ['(a)','(b)','(c)','(d)','(e)', '(f)', '(g)', '(h)']
    rows = int(len(y)/2)
    fig, axs = plt.subplots(nrows=rows, ncols=2,sharex=True, sharey=True)
    
    fig.set_figheight(4)
    fig.set_figwidth(4)
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    axes[2] = '$'+str(axes[2])+'$'
    axes[3] = '$'+str(axes[3])+'$'
    
    for i in range(int(len(y)/2)):
        val = r"$\rm "+labels[i]+"$"
        axs[i,0].plot(x, y[i], colours[i], label=names[i])
        axs[i,0].set_xlabel(axes[0])
        axs[i,0].set_ylabel(axes[1]) 
        axs[i,0].text(-0.22, 1.03, val, transform=axs[i,0].transAxes, size=12, ha='center')
    
    handles, labelss = fig.axes[-1].get_legend_handles_labels()
    fig.legend(names, loc='lower left', mode="expand", ncol=len(names), bbox_to_anchor=(0,1.02,1,0.2))
    for i in range(int(len(y)/2)):
        val = r"$\rm "+labels[i+rows-1]+"$"
        axs[i,1].plot(x, y[i], colours[i])
        axs[i,1].set_xlabel(axes[2])
        axs[i,1].set_ylabel(axes[3]) 
        axs[i,1].text(-0.22, 1.03, val, transform=axs[i,1].transAxes, size=12, ha='center')
        
    fig.tight_layout()
    
    
    if save:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.show() 
    
    
def niceFigure(useLatex=True):
    from matplotlib import rcParams
    plt.rcParams.update({'font.size': 20})
    if useLatex is True:
        plt.rc('text', usetex=True)
        plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    return 
    
"RK4 ODE Solver"
def rk4(diff_eq, y, t, h=0.01):
    k1 = h * diff_eq(y, t)
    k2 = h * diff_eq(y + k1/2, t + h/2)
    k3 = h * diff_eq(y + k2/2, t + h/2)
    k4 = h * diff_eq(y + k3, t + h)
    
    y += (k1 + 2.*k2 + 2.*k3 + k4) / 6.
    return y



"Call ODE Solver"
def ode_solver(f, derivs, time, h=0.01):
    y = np.zeros((len(time),2))
    y[0,:] = np.array([1.0, 0.0])
    
    for i in range(1, len(time)):   # loop over time
        y[i,:] = f(derivs, y[i-1,:], time[i-1], h)
       
    return y


def ode_solver_three_body(f, derivs, time, r, v, h=0.01):
    y = np.asarray([r[0],v[0]])
    yr = np.reshape(y, (12))
    for i in range(1, len(time)):   # loop over time
        yr = np.c_[yr, np.zeros(12)]
        yr[:,i] = f(derivs, yr[:,i-1], time[i-1], h)
       
    return yr

"Leapfrog ODE Solver"
def leapfrog_ode_solver(f, derivs, time, h=0.01):
    r = np.zeros((len(time)))
    r[0] = 1
    v = np.zeros((len(time)))
    
    for i in range(1, len(time)):   # loop over time
        r[i], v[i] = f(derivs, r[i-1], v[i-1], time[i-1], h)
       
    return r, v


def leapfrog_three_body(f, derivs, time, r, v, h=0.01):
    r_new = r.copy()
    v_new = v.copy()
    for i in range(1, len(time)):   # loop over time
        r_new[i], v_new[i] = f(derivs, r_new[i-1], v_new[i-1], time[i-1], h)
       
    return r_new, v_new


"1D Harmonic Oscillator"
def harmonic_oscillator(y,t):
    dy=np.zeros((len(y)))
    dy[0] = y[1]
    dy[1] = -y[0]
    
    return dy


def leapfrog(diffeq, r0, v0, t, h=0.01):
    # About: ...
    hh = h/2.0
    r1 = r0 + hh*diffeq(0, r0, v0, t) # 1: r1 at h/2 using v0
    v1 = v0 + h*diffeq(1, r1, v0, t+hh) # 2: v1 using a(r) at h/2
    r1 = r1 + hh*diffeq(0, r0, v1, t+h) # 3: r1 at h using v1
    
    return r1, v1

def oscillator(id, x, v, t):
    if id == 0:
        return v
    return -w_0**2 * x


def total_energy(x, v):
    return 0.5*v**2 + 0.5*x**2

def energy(r, v):
    # r12, r13, r23 = r[0] - r[1], r[0] - r[2], r[1] - r[2]
    r12v = np.asarray(r[0] - r[1])
    r13v = np.asarray(r[0] - r[2])
    r23v = np.asarray(r[1] - r[2])
    
    s12, s13, s23 = np.sqrt(r12v.dot(r12v)), np.sqrt(r13v.dot(r13v)), np.sqrt(r23v.dot(r23v))
    
    return 0.5*(m1*np.dot(v[0],v[0]) + m2*np.dot(v[1],v[1]) + m3*np.dot(v[2],v[2])) - m1*m2/s12 - m1*m3/s13 - m2*m3/s23


def w0(k, m):
    return math.sqrt(k/m)


def time(initial=0, final=1, h=0.01):
    return np.arange(initial, final*T_0, h)



def animate_q1(x_data_lf, y_data_lf, x_data_rk, y_data_rk, periods, axes, cycle=10, delay=0.1):
    niceFigure(False) 
    
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    
    positions = [[.2, .6, .3, .3], [.6, .6, .3, .3], [.2, .2, .3, .3], [.6, .2, .3, .3]]
    limit = 1.3
    
    fig = plt.figure(figsize=(10,5), dpi=60)
    ax1 = fig.add_axes(positions[0])
    ax1.set_ylim(-limit, limit)
    ax1.set_xlim(-limit, limit)
    ax1.set_title('Leapfrog')
    ax1.set_ylabel('$v [m/s]$')
    ax1.set_xlabel("$x [m/s]$")
    
    ax2 = fig.add_axes(positions[1])
    ax2.set_ylim(-limit, limit)
    ax2.set_xlim(-limit, limit)
    ax2.set_title('Leapfrog')
    ax2.set_ylabel('$v [m/s]$')
    ax2.set_xlabel("$x [m/s]$")
   
    ax3 = fig.add_axes(positions[2])
    ax3.set_ylim(-limit, limit)
    ax3.set_xlim(-limit, limit)
    ax3.set_title('RK4')
    ax3.set_ylabel('$v [m/s]$')
    ax3.set_xlabel('$x [m]$')
    
    ax4 = fig.add_axes(positions[3])
    ax4.set_ylim(-limit, limit)
    ax4.set_xlim(-limit, limit)
    ax4.set_title('RK4')
    ax4.set_ylabel('$v [m/s]$')
    ax4.set_xlabel('$x [m]$')

    line1, = ax1.plot(x_data_lf[0], y_data_lf[0],'xr', markersize=13)
    line2, = ax2.plot(x_data_lf[0], y_data_lf[0],'r-', markersize=13)
    line3, = ax3.plot(x_data_rk[0], y_data_rk[0],'xr', markersize=13)
    line4, = ax4.plot(x_data_rk[0], y_data_rk[0],'r-', markersize=13)
    plt.pause(2) 

    for i in range(len(x_data_lf)):
        if (i % cycle == 0): # very simple animate (update data with pause)
            ax1.set_title("frame time {}".format(i)) # show current time on graph
            ax2.set_title("frame time {}".format(i)) # show current time on graph
            ax3.set_title("frame time {}".format(i)) # show current time on graph
            ax4.set_title("frame time {}".format(i)) # show current time on graph
            
            line1.set_xdata(x_data_lf[i])
            line1.set_ydata(y_data_lf[i])
            line3.set_xdata(x_data_rk[i])
            line3.set_ydata(y_data_rk[i])
            
            line2.set_xdata(x_data_lf[:i])
            line2.set_ydata(y_data_lf[:i])
            line4.set_xdata(x_data_rk[:i])
            line4.set_ydata(y_data_rk[:i])

            plt.draw() # may not be needed (depends on your set up)
            plt.pause(delay) # pause to see animation as code v. fast

      
# Initial values
k, m, w_0 = [1, 1, 1]
m1, m2, m3 = [1, 2, 3]
T_0 = 2*np.pi

#%% 
# Q1(a)

def sho_numerical_plots(periods, h_step):
    h = h_step*T_0
    t = time(final=periods, h=h)

    rk4_result = ode_solver(rk4, harmonic_oscillator, t, h)
    leapfrog_result_dx, leapfrog_result_dv = leapfrog_ode_solver(leapfrog, oscillator, t, h)
    # actual_x = np.cos(w_0*t)
    # actual_v = -w_0*np.sin(w_0*t)

    leapfrog_energy = total_energy(leapfrog_result_dx, leapfrog_result_dv)
    rk4_energy = total_energy(rk4_result[:,0], rk4_result[:,1])

    animate_q1(leapfrog_result_dx, leapfrog_result_dv, rk4_result[:,0], rk4_result[:,1],periods, ["x(m)", "v(m/s)"])

    subplotsax_multi_x(["Leapfrog", "Leapfrog Energy"], [leapfrog_result_dx, t], [leapfrog_result_dv, leapfrog_energy], ["x (m)", "v (m/s)", "t (s)", "E (J)"])
    subplotsax_multi_x(["RK4", "RK4 Energy"], [rk4_result[:,0], t], [rk4_result[:,1], rk4_energy], ["x (m)", "v (m/s)", "t (s)", "E (J)"])


# sho_numerical_plots(40, 0.2)
# sho_numerical_plots(40, 0.02)

#%% 
# Q2
m = [1,2,3]
def collinear_motion_intial_conditions(w):
    euler_quintic = lambda x: x**5 * (m[1]+m[2]) + x**4 * (2*m[1]+3*m[2]) \
                          + x**3 * (m[1]+3*m[2]) - x**2 * (3*m[0]+m[1]) \
                          - x * (3*m[0]+2*m[1]) - (m[0] + m[1])

    lambda_val = optimize.fsolve(euler_quintic, 1)
    lambda_val = lambda_val[0]
    print(lambda_val)

    lambda_val2 = optimize.newton(euler_quintic, 0)
    lambda_val2 = lambda_val2
    print(lambda_val2)

    print(lambda_val - lambda_val2)

    a = (1/w**2 * (m2 + m3 - (m1*(1+2*lambda_val))/(lambda_val**2 * (1+lambda_val)**2)))**(1/3)
    
    x2 = (1/(w**2 * a**2)) * ((m1/lambda_val**2) - m3)
    x1 = x2 - lambda_val*a
    x3 = - (m1*x1 + m2*x2)/m3
    
    x = [x1, x2, x3]
    v_y = [w*x1, w*x2, w*x3]
    
    return x, v_y

initial_x, initial_vy = collinear_motion_intial_conditions(w_0)


def motion(id, r, v, t):
    if id != 0:
        r12v = np.asarray(r[0] - r[1])
        r13v = np.asarray(r[0] - r[2])
        r23v = np.asarray(r[1] - r[2])
        
        s12, s13, s23 = np.sqrt(r12v.dot(r12v)), np.sqrt(r13v.dot(r13v)), np.sqrt(r23v.dot(r23v))
        
        vel = [0]*3
     
        vel[0] = -G*m2*r12v/(s12**3) - G*m3*r13v/s13**3
        vel[1] = G*m1*r12v/(s12**3) - G*m3*r23v/s23**3
        vel[2] = G*m1*r13v/(s13**3) + G*m2*r23v/s23**3
        
        return np.asarray(vel)
        
    return v

def rk4_motion(y, t):
    dy = y.copy()
    
    dy[:6] = dy[6:]  # x = v
    
    r12v = np.asarray(y[:2] - y[2:4])
    r13v = np.asarray(y[:2] - y[4:6])
    r23v = np.asarray(y[2:4] - y[4:6])
    
    s12, s13, s23 = np.sqrt(r12v.dot(r12v)), np.sqrt(r13v.dot(r13v)), np.sqrt(r23v.dot(r23v))
 
    dy[6], dy[7] = -G*m2*r12v/(s12**3) - G*m3*r13v/s13**3
    dy[8], dy[9] = G*m1*r12v/(s12**3) - G*m3*r23v/s23**3
    dy[10], dy[11] = G*m1*r13v/(s13**3) + G*m2*r23v/s23**3
    
    return dy


def three_body(periods, h_step, animate, label="RK4"):
    h = h_step*T_0/w_0
    t = time(final=periods, h=h)
    
    initial_x, initial_vy = collinear_motion_intial_conditions(w_0)
    
    pos = np.zeros([len(t), 3, 2])
    vel = np.zeros([len(t), 3, 2])
    # x, y pairs
    pos[0] = np.array([[initial_x[0], 0], [initial_x[1], 0], [initial_x[2], 0]])
    # vx, vy pairs
    vel[0] = np.array([[0, initial_vy[0]], [0, initial_vy[1]], [0, initial_vy[2]]])
    
    if label == "leapfrog":
        pos_lf, vel_lf = leapfrog_three_body(leapfrog, motion, t, pos, vel, h)
        if animate:
            animate_q2(pos_lf, vel_lf, ["x", "y"], 10)
        else:
            non_animate_q2(pos_lf, vel_lf, ["x", "y"])
            
    elif label == "RK4":
        rk4_result = ode_solver_three_body(rk4, rk4_motion, t, pos, vel, h)
        if animate:
            rk4_animate(rk4_result, ["x", "y"])
        else:
            rk4_non_animate(rk4_result, ["x", "y"])
        
def rk4_non_animate(y, axes):
    niceFigure(False)
    
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    
    fig = plt.figure(figsize=(10,5), dpi=60)
    ax1 = fig.add_axes([.18, .18, .33, .66])
    
    ax1.set_xlabel(axes[0])
    ax1.set_ylabel(axes[1])
    
    lin1, = ax1.plot(y[0][-1], y[1][-1],'or', markersize=13)
    lin2, = ax1.plot(y[2][-1], y[3][-1],'ob', markersize=16)
    lin3, = ax1.plot(y[4][-1], y[5][-1],'om', markersize=19)

    lin4, = ax1.plot(y[0], y[1],'r--', markersize=13)
    lin5, = ax1.plot(y[2], y[3],'b--', markersize=16)
    lin6, = ax1.plot(y[4], y[5],'m--', markersize=19)
    
    plt.draw() 



def rk4_animate(y, axes, cycle=100, delay=0.1):
    niceFigure(False) 
    
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    
    fig = plt.figure(figsize=(10,5), dpi=60)
    ax1 = fig.add_axes([.18, .18, .33, .66])
    ax1.set_ylim(-2,2)
    ax1.set_xlim(-2,2)
    
    ax1.set_xlabel(axes[0])
    ax1.set_ylabel(axes[1])
    
    lin1, = ax1.plot(y[0][0], y[1][0],'or', markersize=13)
    lin2, = ax1.plot(y[2][0], y[3][0],'ob', markersize=16)
    lin3, = ax1.plot(y[4][0], y[5][0],'om', markersize=19)
    
    lin4, = ax1.plot(y[0][0], y[1][0],'r--', markersize=13)
    lin5, = ax1.plot(y[2][0], y[3][0],'b--', markersize=16)
    lin6, = ax1.plot(y[4][0], y[5][0],'m--', markersize=19)
    
    for i in range(y.shape[1]):
        if (i % cycle == 0): # very simple animate (update data with pause)
            ax1.set_title("frame time {}".format(i)) # show current time on graph
            lin1.set_xdata(y[0][i])
            lin1.set_ydata(y[1][i])
            lin2.set_xdata(y[2][i])
            lin2.set_ydata(y[3][i])
            lin3.set_xdata(y[4][i])
            lin3.set_ydata(y[5][i])
            
            lin4.set_xdata(y[0][:i])
            lin4.set_ydata(y[1][:i])
            lin5.set_xdata(y[2][:i])
            lin5.set_ydata(y[3][:i])
            lin6.set_xdata(y[4][:i])
            lin6.set_ydata(y[5][:i])
            
            plt.draw() # may not be needed (depends on your set up)
            plt.pause(delay) # pause to see animation as code v. fast

    
    
def non_animate_q2(r_data, v_data, axes):
    niceFigure(False) 
    
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    
    fig = plt.figure(figsize=(10,5), dpi=60)
    ax1 = fig.add_axes([.18, .18, .33, .66])

    ax1.set_xlabel(axes[0])
    ax1.set_ylabel(axes[1])

    lin1, = ax1.plot(r_data[-1][0][0], r_data[-1][0][1],'or', markersize=13)
    lin2, = ax1.plot(r_data[-1][1][0], r_data[-1][1][1],'ob', markersize=16)
    lin3, = ax1.plot(r_data[-1][2][0], r_data[-1][2][1],'om', markersize=19)
    
    lin4_x = []
    lin4_y = []
    lin5_x = []
    lin5_y = []
    lin6_x = []
    lin6_y = []

    for i in range(len(r_data)): 
        lin4_x.append(r_data[i][0][0])
        lin4_y.append(r_data[i][0][1])
        lin5_x.append(r_data[i][1][0])
        lin5_y.append(r_data[i][1][1])
        lin6_x.append(r_data[i][2][0])
        lin6_y.append(r_data[i][2][1])
        
    
    ax1.plot(lin4_x, lin4_y, 'r-')
    ax1.plot(lin5_x, lin5_y, 'b-')
    ax1.plot(lin6_x, lin6_y, 'm-')
    plt.draw() 



def animate_q2(r_data, v_data, axes, cycle=100, h=0.05, delay=0.1):
    niceFigure(False) 
    
    axes[0] = '$'+str(axes[0])+'$'
    axes[1] = '$'+str(axes[1])+'$'
    
    fig = plt.figure(figsize=(10,5), dpi=60)
    ax1 = fig.add_axes([.18, .18, .33, .66])
    ax1.set_ylim(-2,2)
    ax1.set_xlim(-2,2)

    ax1.set_xlabel(axes[0])
    ax1.set_ylabel(axes[1])

    lin1, = ax1.plot(r_data[0][0][0], r_data[0][0][1],'or', markersize=13)
    lin2, = ax1.plot(r_data[0][1][0], r_data[0][1][1],'ob', markersize=16)
    lin3, = ax1.plot(r_data[0][2][0], r_data[0][2][1],'om', markersize=19)
    
    lin4, = ax1.plot(r_data[0][0][0], r_data[0][0][1],'r--', markersize=13)
    lin5, = ax1.plot(r_data[0][1][0], r_data[0][1][1],'b--', markersize=16)
    lin6, = ax1.plot(r_data[0][2][0], r_data[0][2][1],'m--', markersize=19)
    
    lin4_x = []
    lin4_y = []
    lin5_x = []
    lin5_y = []
    lin6_x = []
    lin6_y = []
    
    plt.pause(2) 

    for i in range(len(r_data)):
        if (i % cycle == 0): # very simple animate (update data with pause)
            ax1.set_title("frame time {}".format(i)) # show current time on graph
            
            lin1.set_xdata(r_data[i][0][0])
            lin1.set_ydata(r_data[i][0][1])
            lin2.set_xdata(r_data[i][1][0])
            lin2.set_ydata(r_data[i][1][1])
            lin3.set_xdata(r_data[i][2][0])
            lin3.set_ydata(r_data[i][2][1])
            
            lin4_x.append(r_data[i][0][0])
            lin4_y.append(r_data[i][0][1])
            lin5_x.append(r_data[i][1][0])
            lin5_y.append(r_data[i][1][1])
            lin6_x.append(r_data[i][2][0])
            lin6_y.append(r_data[i][2][1])
            
            lin4.set_xdata(lin4_x)
            lin4.set_ydata(lin4_y)
            lin5.set_xdata(lin5_x)
            lin5.set_ydata(lin5_y)
            lin6.set_xdata(lin6_x)
            lin6.set_ydata(lin6_y)

            plt.draw() # may not be needed (depends on your set up)
            plt.pause(delay) # pause to see animation as code v. fast
            
            
m1, m2, m3 = [1, 2, 3]
w_0 = 1
G = 1 # 6.67E-11
delta = 10**-9

three_body(4, 0.001, True, "RK4")

# w_0 = 1 + delta
# p, v = three_body(4, 0.01)
# animate_q2(p, ["x", "y"])

# w_0 = 1 - delta
# p, v = three_body(4, 0.01)
# animate_q2(p, ["x", "y"])


    