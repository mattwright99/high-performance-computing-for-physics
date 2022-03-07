from importlib.metadata import PathDistribution
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# plate size, mm
w = h = 20.48
print(f"plate size: {w}x{h} mm")

# grid size
nx = ny = 1024
print(f"grid size:{nx}x{ny}")

# parallel params
chunk = int(ny / size)  # parallelize along y axis

top_pad = 1
if rank == 0:
    top_pad = 0

bottom_pad = 1
if rank == size-1:
    bottom_pad = 0

y_idx_offset = rank * chunk - top_pad

# intervals in x-, y- directions, mm
dx, dy = w/nx, h/ny

# Thermal diffusivity of steel, mm^2/s
D = 4.2
print(f"thermal diffusivity: {D}")

# time
nsteps = 1001
dt = dx**2 * dy**2 / (2 * D * (dx**2 + dy**2))
print(f"dt: {dt}")

# plot
# plot_ts = [0, 40, 90]
plot_ts = [0, 100, 500, 1000]

# array
n_rows = top_pad + chunk + bottom_pad
u = np.zeros((n_rows, nx))

# Initialization - circle osf radius r centred at (cx,cy) (mm)
T_hot = 300
T_cool = 200
r = 5.12
cx = w / 2
cy = h / 2
for i in range(nx):
    for j in range(top_pad, top_pad + chunk):
        p2 = (i*dx - cx)**2 + ((j+y_idx_offset)*dy - cy)**2
        if p2 < r**2:
            radius = np.sqrt(p2)
            u[j, i] = T_cool * np.cos(4*radius)**4
        else:
            u[j, i] = T_hot


def evolve_2d_diff_eq(u):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + D * dt * (
          (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1])/dy**2
        + (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2])/dx**2)
    return u


# Main program
for ts in range(nsteps):
    # communication
    if rank != 0:
        comm.Send(u[1], dest=rank-1, tag=11)
        comm.Recv(u[0], source=rank-1, tag=12)
    
    if rank != size-1:
        comm.Send(u[-2], dest=rank+1, tag=12)
        comm.Recv(u[-1], source=rank+1, tag=11)

    u = evolve_2d_diff_eq(u)

    if ts in plot_ts:
        full_u = None
        if rank == 0:
            full_u= np.empty((nx, ny))
        comm.Gather(u[top_pad : top_pad + chunk], full_u, root=0)
        
        if rank == 0:
            fig = plt.figure(1)
            im = plt.imshow(full_u, cmap=plt.get_cmap('hot'), vmin=T_cool,vmax=T_hot)
            plt.title('{:.1f} ms'.format((ts+1)*dt*1000))
            cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            cbar_ax.set_xlabel('K', labelpad=20)
            fig.colorbar(im, cax=cbar_ax)
            plt.savefig("2d_iter_{}.png".format(ts), dpi=200)
            # plt.show()
            plt.clf()

