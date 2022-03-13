#! /bin/bash

#SBATCH --qos=privileged
#SBATCH --account=teaching
#SBATCH --reservation=teaching
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=1g
#SBATCH --time=00:05:00

cd /global/home/sa118039/high-performance-computing-for-phys/ps4  

time mpirun -n $SLURM_NTASKS python ./parallel_2D_diffusion.py
#time mpirun -n $SLURM_NTASKS python ./com_test.py


