#! /bin/bash

# This script is used to automatically submit jobs with various levels process configurations.

# Fist solve the problem with 1 node and 1, 2, 4, 8 tasks
n_nodes=1
n_tasks=1
for i in {0..3}; do
  let n_tasks=2**i
  echo "submitting $n_nodes node, $n_tasks task job"
  sbatch --nodes=$n_nodes --ntasks=$n_tasks mpi-job.sh
done

# Then solve the problem with 1, 2, 4 nodes and 8 tasks
n_tasks=8
for i in {0..2}; do
  let n_nodes=2**i
  echo "submitting $n_nodes node, $n_tasks task job"
  sbatch --nodes=$n_nodes --ntasks=$n_tasks mpi-job.sh
done

