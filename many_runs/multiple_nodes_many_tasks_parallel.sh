#!/bin/bash
#
# Number of nodes:
#SBATCH --nodes=32
#
#################
# Haswell nodes
#################
#
# Requests Cori Haswell nodes:
#SBATCH --constraint=haswell
#
# Haswell: Assign 1 MPI task to each socket
#SBATCH --tasks-per-node=1
#
#################
# Queue & Job
#################
#
# Which queue to run in: debug, regular, premium, etc. ...
#SBATCH --qos=regular
#
# Run for this much walltime: hh:mm:ss
#SBATCH --time=48:00:00
#
# Use this job name:
#SBATCH -J many1D_lowres
#
# Send notification emails here:
#SBATCH --mail-user=srichers@berkeley.edu
#SBATCH --mail-type=ALL
#
# Which allocation to use:
#SBATCH -A m3761

module load python3
module load parallel
module swap PrgEnv-intel PrgEnv-gnu
NRUNS=1
RUNID=0

ls -d1 i*/j*/k* > input_list.txt

srun --no-kill --ntasks=$SLURM_JOB_NUM_NODES --wait=0 bash payload.sh input_list.txt $NRUNS $RUNID

# test non-slurm command
#bash payload.sh input_list.txt
