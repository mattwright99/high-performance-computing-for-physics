# -*- coding: utf-8 -*-
"""
Time-Dependent Schrodinger Equation using the Space Discretized Leapfrog Technique.

This file contain my code for Problem Set 3 of ENPH 479.

For this problem set, I have designed a TDSE_SpaceDiscretization class that has all the 
funcitonality and attributes needed to run a simulation. Each method is documented and 
the class constructor has a description of the settings the user can provide. The main
methods are the `solve` method which runs the simulation, the `animate_favefn` method
which shows a simulation animation, and the `prob_density` (2D and 3D) variants which
display the PDF as a function of time and position.

Created on Mon Feb 7 15:47:24 2022

@author: Matt Wright
"""


import numpy as np 
import timeit
import matplotlib.pyplot as plt  
from matplotlib import rcParams
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import sparse
import pdb

rcParams['font.size'] = 18
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1


class TDSE_SpaceDiscretization:
    def __init__(
            self,
            spatial_range: list,
            n_spatial_pnts: int,
            t_max: float = None,
            n_time_steps: int = None,
            solver: str = 'slice',
            x0: float = -5,
            sigma: float = 0.5,
            k0: float = 5,
            V: callable = lambda x: np.zeros(x.shape),
            time_dep: bool = False,
        ):
        """Object used to simulate time-dependent Schodinger equation problems using the
        discretized leapfrog method.

        Parameters
        ----------
        spatial_range : list
            Two element list specifying the range in a.u. for which you want to simulate
            the problem over.
        n_spatial_pnts : int
            Number of spatial points to use to discretize the spatial range.
        t_max : float 
            Maximmum time to run the simulation for. Default `None`. User must provide
            either this `t_max` argument or `n_time_steps`.
        n_time_steps : int
            Number of timesteps to simulate over. Default `None`. User must provide
            either `t_max` or this `n_time_steps` argument. Note that time step is computed
            using the spatial step, h, according to: $dt = 0.5 h^2$.
        solver : str
            Method used to solve the ODEs. Can be either `slice`, `sparse`, or `dense`.
            Default is `slice`.
        x0 : float
            Center of the Gaussian used for the intial condition. Default is -5.
        sigma : float
            Width of the Gaussian used for the intial condition. Default is 0.5.
        k0 : float
            Initial average momentum used for the initial condition. Default it 5.
        V : callable
            Function specifying the potential at a given point. Should be vectorized.
            Default is a free particle.
        time_dep : bool
            Whether or not the potential is time-dependent or not. Default is `False`. Set
            to `True` and pass a function that depends on position and space for V. Also,
            note that time dependent potential is only supported for the slicing solver.
        """
        
        self.spatial_range = spatial_range
        self.n_spatial_pnts = n_spatial_pnts
        self.x0 = x0
        self.sigma = sigma
        self.k0 = k0
        self.time_dep = time_dep
        # Check for time dependence and set potential function accordingly. Rest of the code
        # assumes that V has two arguments: x and t
        if time_dep:
            self.V = V
        else:
            # V should depend on both time and space so allow V to accept a time argument
            self.V = lambda x, t: V(x)

        x = np.linspace(spatial_range[0], spatial_range[1], num=self.n_spatial_pnts)  # discretized position
        self.x = np.hstack((x[-1], x, x[0]))  # impose periodic boundary conditions
        self._h = (self.spatial_range[1] - self.spatial_range[0]) / self.n_spatial_pnts  # spatial step size
        self._dt = 0.5 * self._h**2  # time step
        self._b = -0.5 / self._h**2
        self._a = self.V(self.x, 0) - 2 * self._b

        self.set_EOM_solver(solver)
        
        # Set simulation time variables depending on which arguments were passed
        if t_max is None:
            if n_time_steps is None:
                raise Exception('Must specify either `t_max` or `n_time_steps`.')
            self.set_n_time_steps(n_time_steps)
        else:
            self.set_t_max(t_max)

    def solve(self, save_res_f=1):
        """Solve the equations of motion using the defined gaussian wavepacket initial
        conditions and set solver. Returns the spatial evolution and time array. Can
        provie `save_res_f` to specify the fequency at which you want to save your result.
        For example, set too 100 to save the wavefunction 100 time steps (this feature is
        still being tested so you will want to manually set `update_f` if animating result)."""

        R, I = self._intial_conditions()

        n_save_pnts = int(np.ceil(self.n_spatial_pnts/save_res_f))
        result = np.zeros((self.n_time_steps, 2, n_save_pnts))
        t_list = np.arange(0, self.t_max, self._dt)
        
        for i, t in enumerate(t_list):
            R, I = self.leapfrog(R, I, t, self._dt)
            if i % save_res_f == 0:
                result[i, 0, :] = R[1:-1]
                result[i, 1, :] = I[1:-1]
        return result, t_list

    def timed_solver_comparison(self, solvers, n_trials):
        """Times each method to solve the problem and prints average and variance."""

        time_res = np.zeros((len(solvers), n_trials))  # stores results
        for i, solver in enumerate(solvers):
            self.set_EOM_solver(solver)

            for j in  range(n_trials):
                ts = timeit.default_timer()  # start timer
                self.solve()
                time_res[i, j] = timeit.default_timer() - ts
                
            print(solver + ' took: {:.4f} s with varaince: {:.6f}'.format(time_res[i].mean(), time_res[i].var()))

        return time_res

    def animate_wavefn(self, result, t_list, save_name='', update_f=200, T=2*np.pi):
        """Animate the evolution of the wavefunction (specified by `result`) over time."""
        
        rho = self._rho(result[:, 0], result[:, 1])  # probability density function
        V_max = self.V(self.x, 0).max()  # initial max V for normalization

        # Set figure and axis formating
        fig = plt.figure(figsize=(6,6))

        ax = fig.add_axes([.2, .2, .6, .6])
        ax.set_xlabel(r'$x$ (a.u.)')
        ax.set_ylabel(r'$|\psi(x, t)|^2$ (a.u.)')
        ax.set_ylim(-.1, 1.1 * rho.max())
        ax.set_xlim(self.spatial_range[0], self.spatial_range[1])
        # ax.set_title(f'Frame Time: 0')  # used for 1 (a)
        ax.set_title(f'Time ($T_0$): 0.00')
        ax.grid(True, linestyle=':')

        wave_plot, = ax.plot(self.x[1:-1], rho[0])
        if V_max > 0:
            e_plot, = ax.plot(self.x[1:-1], self.V(self.x[1:-1], 0)/50, 'r--', label='$V(x)$')
            fig.text(0.25, 0.7, "$V(x)$", ha='center', linespacing=1.5) 

        plt.pause(2)

        if save_name:  # if name to save the plot is provided (empty string is null)
            plt.savefig(f'{save_name}_0.pdf', dpi=1200, bbox_inches="tight")
            # Define portions of the simulation to saave a snapshot - e.g. 0.3 means snap after 30% of simulation
            # save_indices = (np.array([.3, .7]) * self.n_time_steps / update_f).astype(int)
            save_indices = (np.array([.1, .25, .3]) * self.n_time_steps / update_f).astype(int)
            save_indices = (np.array([.125, .375, .5, .65]) * self.n_time_steps / update_f).astype(int)
        # Animate the PDF evolution and update every `update_f` timesteps
        for c, i in enumerate(range(1, self.n_time_steps, update_f)):
            wave_plot.set_ydata(rho[i])
            if V_max > 0:
                e_plot.set_ydata(self.V(self.x[1:-1], t_list[i])/50)
            # ax.set_title(f'Frame Time: {c}')  # used for 1 (a)
            ax.set_title(f'Time ($T_0$): {round(t_list[i]/T, 2)}')
            plt.draw()

            if save_name and c in save_indices:
                plt.savefig(f'{save_name}_{i}.pdf', dpi=1200, bbox_inches="tight")
            else:
                # since there is breif delay when saving, only pause when not also saving the fig
                plt.pause(0.1)

        if save_name:
            plt.savefig(f'{save_name}_end.pdf', dpi=1200, bbox_inches="tight")

    def prob_density_2D(self, result, t_list, expectation=False, save_name='', T_0=2*np.pi):
        """Display the probability density distribution over time using a 2D plot."""

        # Format figure
        fig = plt.figure(figsize=(6,6))

        ax = fig.add_axes([.2, .2, .6, .6])
        ax.set_ylabel(r'$x$ (a.u.)')
        ax.set_xlabel(r'Time, $t (T_0)$')
        ax.set_ylim(*self.spatial_range)
        
        rho = self._rho(result[:, 0], result[:, 1])
        plt.contourf(t_list/T_0, self.x[1:-1], rho.T)

        if expectation:
            # compute expectation value of X and plot on denstiy graph
            expval_x = np.sum(self._h * self.x[1:-1] * rho, axis=1)
            plt.plot(t_list/T_0, expval_x, 'w--', label=r'$\langle x \rangle (t)$')
            plt.legend()

        if save_name:  # if name to save the plot is provided (empty string is null)
            plt.savefig(f'{save_name}-density.pdf', dpi=1200, bbox_inches="tight")

        plt.show()

    def prob_density_3D(self, result, t_list, save_name='', T_0=2*np.pi):
        """Display the probability density distribution over time using a 2D plot."""
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_xlabel(r'Time, $t (T_0)$')
        ax.set_ylabel(r'$x$ (a.u.)')
        ax.set_zlabel(r'$|\psi(x, t)|^2$ (a.u.)')

        rho = self._rho(result[:, 0], result[:, 1]).T
        x, y = np.meshgrid(t_list/T_0, self.x[1:-1])
        ax.plot_surface(x, y, rho, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        if save_name:  # if name to save the plot is provided (empty string is null)
            plt.savefig(f'{save_name}-density-3d.pdf', dpi=600, bbox_inches="tight")

        plt.show()

    def check_virial(self, result):
        """Checks the Virial Theorem."""

        V = self.V(self.x[1:-1], 0)  # note that this needs to bae altered to handle time-dep V
        rho = self._rho(result[:, 0], result[:, 1])

        # Find derivative of wavefunction
        dR_dx = np.zeros((self.n_time_steps, self.n_spatial_pnts))
        dI_dx = np.zeros((self.n_time_steps, self.n_spatial_pnts))
        for i in range(self.n_spatial_pnts - 1):
            # Use center differencing to estimate derivative of R and I w.r.t X
            dR_dx[:, i] = (result[:, 0, i+1] - result[:, 0, i-1]) / (2*self._h)
            dI_dx[:, i] = (result[:, 1, i+1] - result[:, 1, i-1]) / (2*self._h)
        
        # NOTE: since it is bounded in the well the edge derivatives are negligible
        dRho_dx = self._rho(dR_dx, dI_dx)

        # Perform integration over all space and average over time 
        expval_T = 0.5 * np.sum(self._h * dRho_dx) / self.n_time_steps
        expval_V = np.sum(self._h * V * rho) / self.n_time_steps

        print(f'2<T> = {2 * round(expval_T, 2)}')
        print(f'2<V> = {2 * round(expval_V, 2)}')

    def _intial_conditions(self):
        """Initial wavepacket condition. Gaussian with center at x0, width of sigma, and
        initial average momentum of k0 evaluated over all x. These parameters specifying
        the distribution are attributes of the object."""

        gauss = (self.sigma * np.sqrt(np.pi))**(-1/2) * np.exp(-1 * (self.x - self.x0)**2 / (2 * self.sigma**2))
        R = gauss * np.cos(self.k0 * self.x)
        I = gauss * np.sin(self.k0 * self.x)
        return R, I

    def leapfrog(self, r_0 , v_0 , t, h):
        """Vectorized leapfrog method using numpy arrays.
        
        Parameters
        ----------
        self : object
            Object pointing to self that must have an attribute `diff_eq` which is a
            callable of the form f(id, r, v, t) and computes the equations of motion of the
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

        r_12 = r_0 + h/2 * self.diff_eq(0, r_0, v_0, t)    # r_{1/2} at h/2 using v0=dx/dt
        v_1 = v_0 + h * self.diff_eq(1, r_12, v_0, t+h/2)  # v_1 using a(r) at h/2
        r_1 = r_12 + h/2 * self.diff_eq(0, r_0, v_1, t+h)  # r_1 at h using v_1
        return r_1, v_1

    def set_EOM_solver(self, solver):
        """Set the method used to solve the equations of motion. Supported optins are
        `slice`, `sparse`, and `dense`."""

        if solver == 'slice':
            self.diff_eq =  self._eqns_of_motion_slice
            return

        # Set diff eq solver to matrix-based EOMs
        self.diff_eq = self._eqns_of_motion_matrix

        # Build 'A' matrix
        b = self._b * np.ones(self.n_spatial_pnts+1)
        mat = np.diag(self._a)
        mat += np.diag(b, k=1)
        mat += np.diag(b, k=-1)
        if solver == 'sparse':
            self.A = sparse.csc_matrix(mat)  # convert to sparse
        elif solver == 'dense':
            self.A = mat
        else:
            self.diff_eq = None
            raise Exception(f'ERROR: Invalid solver provided. Must be `slice`, `sparse`, `dense` but got: {solver}')

    def set_n_time_steps(self, n):
        """Set the number of time steps."""

        self.n_time_steps = n
        self.t_max = n * self._dt

    def set_t_max(self, t_max):
        """Set the time of simulation."""

        self.t_max = t_max
        self.n_time_steps = int(np.ceil(t_max / self._dt))

    def _eqns_of_motion_slice(self, id, R, I, t):
        """Use slicing to exploit the tridiagonal form of the matrix that defines system
        evolution for efficient matrix multiplication."""

        if self.time_dep:
            # Compute current potential if time dependent
            self._a = self.V(self.x, t) - 2*self._b

        if id == 0:
            dR_dt = self._a * I  # compute elements along the diagonal
            dR_dt[1 : -1] += self._b * (I[:-2] + I[2:])  # add the lower and upper off diagonal components
            # Impose periodic boundary conditions
            dR_dt[0] = dR_dt[-2]
            dR_dt[-1] = dR_dt[1]
            return dR_dt
        else:
            dI_dt = self._a * R
            dI_dt[1 : -1] += self._b * (R[:-2] + R[2:])  # add the lower and upper off diagonal components
            # Boundary conditions
            dI_dt[0] = dI_dt[-2]
            dI_dt[-1] = dI_dt[1]
            return -1 * dI_dt

    def _eqns_of_motion_matrix(self, id, R, I, t):
        """Use matrix based multiplication. Dense or sparse depending on if solver was set
        to `sparse` or `dense`. Note that time-dependent potential for these approachs
        requires too much overhead so it is only supported for the `slice` solver."""

        if id == 0:
            dR_dt = self.A @ I
            dR_dt[0] = dR_dt[-2]
            dR_dt[-1] = dR_dt[1]
            return dR_dt
        else:
            dI_dt = self.A @ R
            dI_dt[0] = dI_dt[-2]
            dI_dt[-1] = dI_dt[1]
            return -1 * dI_dt

    def _rho(self, R, I):
        return R**2 + I**2


#%% Question 1

print('\nStarting Question 1...')

tdse = TDSE_SpaceDiscretization(
    spatial_range=[-10, 10],
    n_spatial_pnts=1000,
    n_time_steps=15000,
    solver='slice',
)
res, tlist = tdse.solve()
tdse.animate_wavefn(res, tlist)


# Part (b)

print('Starting Question 1(b)...')


n_time_runs = 3  # number of timed runs to average over for each solver
solvers = ['slice', 'sparse', 'dense']  # solvers to test

tdse = TDSE_SpaceDiscretization(
    spatial_range=[-10, 10],
    n_spatial_pnts=1000,
    n_time_steps=1000,
)
print('\n[Q1]: Solver comparison with 1000 time steps and spatial points:')
tdse.timed_solver_comparison(solvers, n_time_runs)

# Now try with 2000 spatial points
tdse = TDSE_SpaceDiscretization(
    spatial_range=[-10, 10],
    n_spatial_pnts=2000,
    n_time_steps=1000,
)
print('\n[Q1]: Solver comparison with 2000 spatial points:')
tdse.timed_solver_comparison(solvers, n_time_runs)

plt.close('all')
del res


#%% Question 2

print('\nStarting Question 2...')

T = 2 * np.pi

tdse = TDSE_SpaceDiscretization(
    spatial_range=[-10, 10],
    n_spatial_pnts=1000,
    t_max=2*T,
    solver='slice',
    k0=0,
    V=lambda x: 0.5 * x**2
)
res, tlist = tdse.solve()

tdse.animate_wavefn(res, tlist)
tdse.prob_density_2D(res[::200], tlist[::200])  # NOTE: uses alot of space so I downsample with the slicing
# tdse.prob_density_2D(res, tlist)  # uncomment to not down sample
# tdse.prob_density_3D(res[::200], tlist[::200])  # Commented out since it can computationally demmanding

print('\n[Q2]: Virial Theorem Results:')
tdse.check_virial(res)

plt.close('all')
del res


#%% Question 3

print('\nStarting Question 3...')

def potential(x):
    return a * x**4 - b * x**2

a = 1
b = 4

T = 2 * np.pi

tdse = TDSE_SpaceDiscretization(
    spatial_range=[-5, 5],
    n_spatial_pnts=500,
    t_max=4*T,
    solver='slice',
    x0=-np.sqrt(2),
    sigma=0.5,
    k0=0,
    V=potential
)
res, tlist = tdse.solve()

tdse.animate_wavefn(res, tlist)
tdse.prob_density_2D(res[::200], tlist[::200], expectation=True)


a = 1
b = 2
print('[Q3]: Using a=1, b=2')

tdse = TDSE_SpaceDiscretization(
    spatial_range=[-5, 5],
    n_spatial_pnts=500,
    t_max=4*T,
    solver='slice',
    x0=-np.sqrt(2),
    sigma=0.5,
    k0=0,
    V=potential
)
res, tlist = tdse.solve()
tdse.prob_density_2D(res[::200], tlist[::200], expectation=True)


a = 1
b = 8
print('[Q3]: Using a=1, b=8')

tdse = TDSE_SpaceDiscretization(
    spatial_range=[-5, 5],
    n_spatial_pnts=500,
    t_max=4*T,
    solver='slice',
    x0=-np.sqrt(2),
    sigma=0.5,
    k0=0,
    V=potential
)
res, tlist = tdse.solve()
tdse.prob_density_2D(res[::200], tlist[::200], expectation=True)

plt.close('all')
del res

