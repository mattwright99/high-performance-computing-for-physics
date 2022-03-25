# -*- coding: utf-8 -*-
"""
Parallelized 2D Heat Equation

This file contain my plotting code for Problem Set 4 of ENPH 479.

I ran a parallelized simulation of the 2D heat equation using the
``parallel_2D_diffusion.py`` file on the CAC Frontenac cluster and saved the snap shots of
the simulation at various time steps. This script is used to nicely visulaize the data.

Created on Sun Mar 13

@author: Matt Wright
"""

import numpy as np
import os
from matplotlib import cm, colors, rcParams
import matplotlib.pyplot as plt

rcParams['font.size'] = 18
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['xtick.major.width'] = 1
rcParams['ytick.major.width'] = 1

# Initial temperature conditions set in simulation
vmin = 300
vmax = 2000

# Iterate over all results and plot
n_files = len(os.listdir('./results/'))
for i in range(n_files):
    # Load full temperature data
    u = np.load(f'./results/2d_u_{i*100}.npy')

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.2,.2,.6,.6])

    im = ax.imshow(u, cmap=plt.get_cmap('hot'), vmin=vmin,vmax=vmax)
    ax.set_title(f'Timestep: {i*100}')
    ax.set_xlabel('$x$ (mm)')
    ax.set_ylabel('$y$ (mm)')
    ax.set_xticks([])
    ax.set_yticks([])

    # plt.savefig("./figs/2d_iter_{}.pdf".format(i*100), dpi=800)

# Generate a plot of just the color bar used for the density plots - see Fig 1 in report
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
