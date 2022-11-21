#!/bin/bash
#
# Number of nodes:
#SBATCH --nodes=4
#
#################
# Haswell nodes
#################
#
# Requests Cori Haswell nodes:
#SBATCH --constraint=haswell
#
# Haswell: Assign 1 MPI task to each socket
#SBATCH --tasks-per-node=16
#
# Haswell: each socket has 32 CPUs (with hyperthreading)
#SBATCH --cpus-per-task=4
#
#################
# Queue & Job
#################
#
# Which queue to run in: debug, regular, premium, etc. ...
#SBATCH --qos=regular
#
# Run for this much walltime: hh:mm:ss
#SBATCH --time=04:00:00
#
# Use this job name:
#SBATCH -J flash_reduce
#
# Send notification emails here:
#SBATCH --mail-user=ebgrohs@ncsu.edu
#SBATCH --mail-type=ALL
#
# Which allocation to use:
#SBATCH -A m3761

#SBATCH -o stdout.out
#SBATCH -e stdout.out
#SBATCH --mem-per-cpu=1500

# On the compute node, change to the directory we submitted from
cd $SLURM_SUBMIT_DIR

source ~/.mymods
source ~/.myomp

source activate myenv

date

mpirun -n 64 python ~/Emu_scripts/data_reduction/reduce_data.py

date
