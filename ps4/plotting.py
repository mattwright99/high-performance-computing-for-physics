import numpy as np
import os
from matplotlib import cm, colors, rcParams
import matplotlib.pyplot as plt

rcParams['font.size'] = 18
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1

vmin = 300
vmax = 2000
n_files = len(os.listdir('./results/'))
for i in range(n_files):
    u = np.load(f'./results/2d_u_{i*100}.npy')

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.2,.2,.6,.6])

    im = ax.imshow(u, cmap=plt.get_cmap('hot'), vmin=vmin,vmax=vmax)
    ax.set_title(f'Timestep: {i*100}')
    ax.set_xlabel('$x$ (mm)')
    ax.set_ylabel('$y$ (mm)')

    ax.set_xticks([0, 500, 1000])
    ax.set_yticks([0, 500, 1000])

    # plt.savefig("./figs/2d_iter_{}.pdf".format(i*100), dpi=800)


rcParams['font.size'] = 8
fig = plt.figure(figsize=(8,1))

cbar_ax = fig.add_axes([0.1, 0.5, 0.8, 0.2])
cbar_ax.set_title('Temperature (K)', fontsize=8)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
plt.colorbar(
    cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('hot')),
    cax=cbar_ax, orientation='horizontal', ticks=[400, 800, 1200, 1600, 2000]
)
# plt.savefig("./figs/colorbar.pdf", dpi=800)
plt.show()
