import numpy as np
import matplotlib.pyplot as plt

# plate size, mm
w = h = 20.48
print("plate size:",w,"x",h,"mm")

# grid size
nx = ny = 1024
print("grid size:",nx,"x",ny)

# intervals in x-, y- directions, mm
dx, dy = w/nx, h/ny
dx2, dy2 = dx*dx, dy*dy

# Thermal diffusivity of steel, mm2/s
D = 4.2
print("thermal diffusivity:",D)

# time
nsteps = 1001
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
print("dt:",dt)

# plot
plot_ts = [0, 100, 500, 1000]

# array
u0 = np.zeros((ny, nx))
# u = u0.copy()

# Initialization - circle of radius r centred at (cx,cy) (mm)
Tcool, Thot = 200, 2000
r, cx, cy = 5.12, w/2, h/2
r2 = r**2
for i in range(nx):
    for j in range(ny):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        if p2 < r2:
            radius = np.sqrt(p2)
            u0[j, i] = Thot * np.cos(4*radius)**4
        else:
            u0[j, i] = Tcool


def do_timestep(u0):
    # Propagate with forward-difference in time, central-difference in space
    u0[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
        (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dy2
        + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dx2 )
    return u0

# Output figures at the chosen timesteps
for m in range(nsteps):
    u0 = do_timestep(u0)
    if m in plot_ts:
        fig = plt.figure(1)
        im = plt.imshow(u0, cmap=plt.get_cmap('hot'), vmin=Tcool,vmax=Thot)
        plt.title('{:.1f} ms'.format((m+1)*dt*1000))
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar_ax.set_xlabel('K', labelpad=20)
        fig.colorbar(im, cax=cbar_ax)
        # plt.savefig("iter_{}.png".format(m), dpi=200)
        plt.show()
        # plt.clf()
