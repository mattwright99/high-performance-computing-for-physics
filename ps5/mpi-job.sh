#! /bin/bash

#SBATCH --qos=privileged
#SBATCH --account=teaching
#SBATCH --reservation=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=01:00:00

echo "Running $SLURM_NNODES nodes and $SLURM_NTASKS tasks..."

# change to source code directory
cd ~/high-performance-computing-for-phys/ps5

time mpirun -n $SLURM_NTASKS python ./2d_parallel_FDTD.py
