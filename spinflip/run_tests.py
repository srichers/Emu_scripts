import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../data_reduction")
import numpy as np
import spin_flip_tools as sft
from four_current import store_gradients

# For ease of development, reload the relevant modules to make sure they are up to date.
import importlib
importlib.reload(sft)

#location=[80,73,99]
#xmin = 75
#xmax = 83
#ymin = 67
#ymax = 77
#zmin = 97
#zmax = 99
location=[100,120,99]
xmin = 94
xmax = 132
ymin = 119
ymax = 138
zmin = 97
zmax = 99

merger_data_filename = "/mnt/scratch/shared/2-orthonormal_distributions/model_rl0_orthonormal_rotated.h5"
emu_data_loc = "/mnt/scratch/shared/3-Henry_NSM_box/"
emu_filename = emu_data_loc + "i{:03d}".format(location[0])+"_j{:03d}".format(location[1])+"_k{:03d}".format(location[2])+"/allData.h5"
rangestring = "_i"+str(xmin)+"-"+str(xmax)+"_j"+str(ymin)+"-"+str(ymax)+"_k"+str(zmin)+"-"+str(zmax)
gradient_filename_start = "/mnt/scratch/shared/4-gradients/gradients_start"+rangestring+".h5"
gradient_filename_end = "/mnt/scratch/shared/4-gradients/gradients_end"+rangestring+".h5"
p_abs = 1e7 # eV

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
#store_gradients(merger_data_filename, emu_data_loc, gradient_filename_start, xmin, xmax, ymin, ymax, zmin, zmax, 0)
#store_gradients(merger_data_filename, emu_data_loc, gradient_filename_end, xmin, xmax, ymin, ymax, zmin, zmax, -1)
