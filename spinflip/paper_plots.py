import os
import sys
sys.path.append("/mnt/scratch/henrypurcell/N3AS/Emu_scripts/spinflip")
import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
from constants import c, M_p, M_3flavor, G, hbar
import spin_flip_tools as sft
import gellmann as gm
import four_current as fc
import matrix
import diagonalizer as dg
import time_plots as tp
import merger_grid as mg
import glob
import concurrent
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from matplotlib import cm, colors
import matplotlib.axes as ax
from mpl_toolkits.mplot3d import Axes3D
from four_current import four_current, read_gradients
from matrix import visualizer
#import latex
from itertools import product

################
# plot options #
################
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['xtick.top'] = True

mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['ytick.right'] = True

mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['axes.grid'] = False


#####################
###plot parameters###
#####################
merger_data_filename = "/mnt/scratch/shared/2-orthonormal_distributions/model_rl0_orthonormal_rotated.h5"
emu_data_loc = "/mnt/scratch/shared/3-Henry_NSM_box/"
gradient_filename = "/mnt/scratch/shared/4-gradients/gradients_start.h5"

#3 z-values of cross sections to plot
zvals = [97,98,99]

#location of specific point to analyze
location=[80,73,zvals[1]] #will appear in middle plot in contour_plot hence zvals[1]
emu_filename = emu_data_loc + "i{:03d}".format(location[0])+"_j{:03d}".format(location[1])+"_k{:03d}".format(location[2])+"/allData.h5"

#boundaries of regions over which gradients have been computed (for adiabaticity analysis)
#nth item in list corresponds to nth cross section (currently all are the same)
xmins = [75,75,75]
xmaxs = [83,83,83]
ymins = [67,67,67]
ymaxs = [77,77,77]

#neutrino momentum
p_abs = 1e7 # eV



###########################################
### PART 1: INITIAL STATE OF THE MERGER ###
###########################################

### Resonance and FFI Regions Plot: plots resonance and FFI regions on z-cross sections ###
mg.Merger_Grid(zvals, merger_data_loc=merger_data_filename, p_abs=p_abs
               ).contour_plot(x=[None,location[0],None], 
                              y=[None,location[1],None], 
                              rect_xmin = xmins, rect_xmax = xmaxs,
                              rect_ymin = ymins, rect_ymax = ymaxs,
                              savefig = 'merger_grid_t0')

### Adiabaticity Plot: plots average value of adiabaticity on resonance band at each grid cell###
#(averaging over azimuthal anglues, picking phi_resolution of these at each cell)
#Plots for z-cross sections - first input is a list of z's to plot. We can do 3 cross sections for the paper
sft.Gradients(gradient_filename, merger_data_filename
              ).plotAdiabaticities(zvals, emu_data_loc, p_abs, 1E-8, 1E-7,
                                   savefig = 'adiabaticity_grid_t0')



######################################
### PART 2: 3-FLAVOR, 3D SPIN-FLIP ###
######################################

### Angular plot of Eigenvector Resonance value (1-|L-R|) and zoomed in version of the same ###
SP_0 = sft.SpinParams(0, emu_filename, merger_data_filename, location, p_abs, resonance_type = 'simplified', gradient_filename = gradient_filename)
SP_0.angularEigenvectorPlot(200, 200,
                             value = 'lminusr',
                             phi_optimal = 3/4*np.pi, zoom = 0.1, shift = [-0.05,0],
                             vmax = -4,
                             zoom_resolution = 200, initvector = 'default', 
                             method = 'Nelder-Mead', bounds =[(np.pi/4, 3*np.pi/4)], 
                             savefig='angular_eigenvector_plot_t0',  linearPlot = False)



### Linear plot of Eigenvector resonance value vs theta for phi = 3/4 pi (shows 1D phi-slice of zoomed in picture from previous plot)###
# this identifies resonance bands in the channels given in flavor-resonances with the given colors. There are 9 resonance bands total, and we can show 3
# after showing 3 resonance bands we can plot their evolution in the next plot to show they are resonant.
# function outputs the relevant resonant thetas for use in the next plot
resonant_thetas = SP_0.linearEigenvectorPlot(300,  
                              initvector = None, value = 'lminusr',
                              zoom = 0.1, shift = -0.05, phi_optimal= np.pi,
                              method = 'Nelder-Mead',
                              extra_lines = None, extra_init_vectors = None,
                              flavor_resonances = [(0,0,'cyan'), (1,1,'lime'), (0,1,'magenta')],
                              savefig = 'linear_eigenvector_plot_t0')

#sometimes scipy fails to find the exactly correct theta value for super narrow bands, 
#in which case we can manually input the correct value which can be found by reducing the search bounds (needs to be done manually)
#in this example case, the third angle is wrong
resonant_thetas[2] = 1.3633226584851739
  
    
### Diagnonalizer plot for identified resonance bands: shows evolution of resonant state for the resonant hamiltonians identified in the last plot







### Single-Point Adiabaticity plot: shows adiabaticity and gradient vs azimuthal angle over resonance band for selected point ###
SP_0.azimuthalGradientsPlot(phi_resolution = 300,  vmax = 5,
                            savefig = 'azimuthal_gradients_t0')

# Adiabaticity width for single-point: prints and returns the angular width of the adiabatic region on the resonance band at the given point
# We'll quote this value when describing the plot
SP_0.findAdiabaticRegions(phi_resolution = 50, min_dist_between_peaks = 10,
                                          adiabaticity_threshold = 1, max_peak_count = 3,
                                          savefig=False, vmax = 1E-5, method = 'Nelder-Mead')