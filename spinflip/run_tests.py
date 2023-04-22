import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import spin_flip_tools as sft

# For ease of development, reload the relevant modules to make sure they are up to date.
import importlib
importlib.reload(sft)

data_loc = "/mnt/scratch/shared/2-orthonormal_distributions/model_rl0_orthonormal_unrotated.h5"

##########
# STEP 1 #
##########
# basic merger data located in 1-Francois_data

##########
# STEP 2 #
##########
# Generate the orthonormal distribution file in 2-orthonormal_distributions
# python3 orthonormal_distributions.py
sft.Merger_Grid(zval=98, data_loc=data_loc).contour_plot(savefig = True)

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
#process simulation data from a dataset (inputdatafile) full of files of the form i*j*k*/allData.h5 (e.g. Henry_NSM_Box)
#outputs h5 files in directory outputpath
#sft.Multipoint_interact("/mnt/scratch/shared/3-Henry_NSM_box", "/mnt/scratch/shared/4-Multipoint_interact/test").run_many()

##########
# STEP 5 #
##########
# For one grid cell, calculate all spin transformation quantities at each timestep
sft.SpinParams(t_sim = 100,
               data_loc='/mnt/scratch/shared/4-Multipoint_interact/i077_j070_k097_sfmJ.h5',
               merger_data_loc=data_loc,
               location=[77,70,97]).angularPlot(100,100)

# Draw adiabaticity/resonance for many points
# Draw angular distribution at one point
# Draw diagonalizer sinusoidal distribution
# Draw Hamiltonian matrix
sft.Multipoint(80,73,99,"/mnt/scratch/shared/4-Multipoint_interact", 75,80,73,78,data_loc).pointPlots(0,savefig=True)


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
sft.Diagonalizer( H= np.array([[0,1],[1,0]])).state_evolution_plotter(init_array = np.diag((1,0)))
