#! /bin/bash

#SBATCH --qos=privileged
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --time=00:10:00

cd /global/home/sa118039/high-performance-computing-for-phys/ps4  

time mpirun -n $SLURM_NTASKS python ./parallel_1D_diffusion.py

