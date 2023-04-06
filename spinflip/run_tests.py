import os
import sys
sys.path.append("/mnt/scratch/srichers/software/emu_scripts/data_reduction")
import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy
from scipy import optimize as opt
import h5py
import amrex_plot_tools as amrex
import emu_yt_module as emu
import spin_flip_tools as sft
import glob
import concurrent
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from matplotlib import cm, colors
import matplotlib.axes as ax
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

# For ease of development, reload the relevant modules to make sure they are up to date.
import importlib
importlib.reload(sft)

c = 299792458 #m/s
hbar =6.582119569E-16 #eV s
G=1.1663787E-23 # eV^-2 (fermi constant)
M_p=1.6726219*10**(-24)#grams (Proton mass)

################
# plot options #
################
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
COLOR = 'black'
mpl.rcParams['text.color'] = COLOR
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rc('text', usetex=True)
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.grid'] = False
plt.show()
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'

Htest= np.zeros((6,6))
Htest[3,0]=1
Htest[0,3]=1

Htest_2f= np.zeros((4,4))
Htest_2f[2,0]=1
Htest_2f[0,2]=1

sft.Diagonalizer(H = Htest).state_evolution_plotter(init_array = np.diag((1,0,0,0,0,0)))
sft.Diagonalizer(H=Htest_2f).state_evolution_plotter(init_array = np.diag((1,0,0,0)))
sft.Diagonalizer( H= np.array([[0,1],[1,0]])).state_evolution_plotter(init_array = np.diag((1,0)))

sft.SpinParams(t_sim = 100, data_loc='/mnt/scratch/shared/spinflip/sfm/i106_j136_k099_sfm_JJ.h5', merger_data_loc="/mnt/scratch/shared/spinflip/merger_grid.h5").angularPlot(100,100)


Multipoint(80,73,99).pointPlots(0,1E-8)

Merger_Grid().contour_plot()
