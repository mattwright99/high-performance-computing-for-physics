#! /bin/bash

n_nodes=1
n_tasks=1
for i in {0..3}; do
  let n_tasks=2**i
  echo "submitting $n_nodes node, $n_tasks task job"
  sbatch --nodes=$n_nodes --ntasks=$n_tasks mpi-job.sh
done

n_tasks=8
for i in {0..2}; do
  let n_nodes=2**i
  echo "submitting $n_nodes node, $n_tasks task job"
  sbatch --nodes=$n_nodes --ntasks=$n_tasks mpi-job.sh
done

