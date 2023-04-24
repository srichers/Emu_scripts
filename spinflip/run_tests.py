import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import spin_flip_tools as sft

# For ease of development, reload the relevant modules to make sure they are up to date.
import importlib
importlib.reload(sft)

location=[80,73,99]

merger_data_filename = "/mnt/scratch/shared/2-orthonormal_distributions/model_rl0_orthonormal_unrotated.h5"
emu_data_loc = "/mnt/scratch/shared/3-Henry_NSM_box/"
emu_filename = emu_data_loc + "i{:03d}".format(location[0])+"_j{:03d}".format(location[1])+"_k{:03d}".format(location[2])+"/allData.h5"

##########
# STEP 1 #
##########
# basic merger data located in 1-Francois_data

##########
# STEP 2 #
##########
# Generate the orthonormal distribution file in 2-orthonormal_distributions
# python3 orthonormal_distributions.py

##########
# STEP 3 #
##########
# Run Emu simulations in 3-Henry_NSM_box
# Henry specifies i,j,k range
# python3 setup_runs.sh
# bash run_all.sh

##########
# STEP 4 #
##########
# Calculate gradients

##########
# STEP 5 #
##########
# Draw adiabaticity/resonance for many points
# Draw angular distribution at one point
# Draw diagonalizer sinusoidal distribution
# Draw Hamiltonian matrix
sft.MultiPlot(location[0], location[1], location[2], emu_filename, 75,80,73,78,merger_data_filename).pointPlots(0,savefig=True)

######################
# Diagonalizer Tests #
######################
Htest= np.zeros((6,6))
Htest[3,0]=1
Htest[0,3]=1

Htest_2f= np.zeros((4,4))
Htest_2f[2,0]=1
Htest_2f[0,2]=1

sft.Diagonalizer(H = Htest).state_evolution_plotter(init_array = np.diag((1,0,0,0,0,0)))
sft.Diagonalizer(H=Htest_2f).state_evolution_plotter(init_array = np.diag((1,0,0,0)))
sft.Diagonalizer(H= np.array([[0,1],[1,0]])).state_evolution_plotter(init_array = np.diag((1,0)))
