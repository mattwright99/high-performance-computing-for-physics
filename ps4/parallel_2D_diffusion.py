from importlib.metadata import PathDistribution
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# plate size, mm
w = h = 20.48
print(f"[P{rank}] plate size: {w}x{h} mm")

# grid size
nx = ny = 1024
print(f"[P{rank}] grid size:{nx}x{ny}")

# parallel params
chunk = int(ny / size)  # parallelize along y axis

y_idx_offset = rank * chunk

# intervals in x-, y- directions, mm
dx, dy = w/nx, h/ny

# Thermal diffusivity of steel, mm^2/s
D = 4.2
print(f"[P{rank}] thermal diffusivity: {D}")

# time
nsteps = 1001
dt = dx**2 * dy**2 / (2 * D * (dx**2 + dy**2))
print(f"[P{rank}] dt: {dt}")

# plot
plot_ts = np.arange(0, 1001, 100, dtype=int)

# array
u = np.zeros((chunk+2, nx), dtype=np.float64)

# Initialization - circle osf radius r centred at (cx,cy) (mm)
T_hot = 2000
T_cool = 300
r = 5.12
cx = w / 2
cy = h / 2
for i in range(nx):
    for j in range(chunk):
        row_i = j + 1 # row index offset by 1 for padding
        p2 = (i*dx - cx)**2 + ((j+y_idx_offset)*dy - cy)**2
        if p2 < r**2:
            radius = np.sqrt(p2)
            u[row_i, i] = T_hot * np.cos(4*radius)**4
        else:
            u[row_i, i] = T_cool


def evolve_2d_diff_eq(u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + D * dt * (
          (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
        + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2)
    return u


print(f'[P{rank}] Running for {nsteps} time steps...')

# Main program
for ts in range(nsteps):
    # communication
    if rank != 0:
        comm.Send(u[1], dest=rank-1, tag=11)
        comm.Recv(u[0], source=rank-1, tag=12)
        
    if rank != size-1:
        comm.Recv(u[-1], source=rank+1, tag=11)
        comm.Send(u[-2], dest=rank+1, tag=12)

    u = evolve_2d_diff_eq(u)

    if ts in plot_ts:
        full_u = None
        if rank == 0:
            full_u= np.empty((nx, ny))
        comm.Gather(u[1 : chunk+1], full_u, root=0)
        
        if rank == 0:
            np.save(f'2d_u_{ts}.npy', full_u)

            fig = plt.figure(figsize=(6,6))
            ax = fig.add_axes([0.2,.2,.6,.6])

            im = ax.imshow(full_u, cmap=plt.get_cmap('hot'), vmin=T_cool,vmax=T_hot)
            ax.set_title('{:.1f} ms'.format((ts+1)*dt*1000))
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')

            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
            cbar_ax.set_xlabel('K', labelpad=20)
            fig.colorbar(im, cax=cbar_ax)
            
            plt.savefig("big_2d_iter_{}.png".format(ts), dpi=100)
            plt.clf()

