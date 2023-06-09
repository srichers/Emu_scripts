# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#2_3:
n_nue0 = 4.89e+32     # 1/ccm
#n_nux0 = 0.0 # 1/ccm
n_nux0 = 4.89e+27     # 1/ccm
mfact = 1.e-32
mfactstr = r'$10^{-32}\times$'

#NSM_1:
#n_nue0 = 1.421954234999705e+33     # 1/ccm
#n_nux0 = 1.9645407875568215e+33/4. # 1/ccm, each flavor
#NSM_2:
#n_nue0 = 2.3293607911671233e+33    # 1/ccm
#n_nux0 = 1.5026785300973756e+33 # 1/ccm, each flavor
#mfact = 1.e-33
#mfactstr = r'$10^{-33}\times$'

n_tot = n_nue0 + 2.*n_nux0
n_2F = n_nue0 + n_nux0
n_tot_eq = n_tot/3.0
n_2F_eq = n_2F/2.0


################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
#mpl.rcParams['text.usetex'] = True
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


#############
# read data #
#############
if(len(sys.argv) != 2):
    print()
    print("Usage: [Nex_comps_vs_time.py filename], where filename is the .dat file to read from")
    print()
    exit()

filename = sys.argv[1]
print("Using {} for input".format(filename))

fin = open(filename, 'r')
fin_lines = fin.readlines()
fin.close()

num_cells = int(fin_lines[0].split()[-1])
print("Number of cells = ", num_cells)
num_steps = int(fin_lines[1].split()[-1])
print("Number of time steps = ", num_steps)
fin_lines = fin_lines[2:]

block_len = num_steps + 2

t_array = np.empty((num_steps))
nex_array = np.empty((num_steps,4))

for i in range(num_cells):
    start_ind = i*block_len
    cell_id = int(fin_lines[start_ind].split()[-1])
    print('Working on Cell = ', cell_id)

    j1 = 0
    for j in range(start_ind+1,start_ind+block_len,1):
        for k,word in enumerate(fin_lines[j].split()):
            if k == 0 and i == 0:
                t_array[j1] = float(word)*1.e+09
            if k == 3:
                nex_array[j1,0] = float(word)
            if k == 4:
                nex_array[j1,1] = float(word)
            if k == 7:
                nex_array[j1,2] = float(word)
            if k == 8:
                nex_array[j1,3] = float(word)
        j1 += 1

    fig, axes = plt.subplots(2,1, figsize=(6,10), sharex=True)
    plt.subplots_adjust(hspace=0)

    ##############
    # formatting #
    ##############
    axes[1].set_xlabel(r'$t\,(10^{-9}\,{\rm s})$')
    for i in range(2):
        axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
        axes[i].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i].minorticks_on()
    #axes.set_xlim(-1.0, 4.0)
    #axes.set_xlim(-0.5, 2.0)
    #axes.set_xlim(-0.5, 1.0)
    #axes.set_xlim(-0.7, 0.5)
    #axes.set_ylabel(mfactstr + r'$|N_{ex}|\,({\rm cm}^{-3})$')
    #axes.set_ylim(0.9*mfact*n_nux0, 1.1*mfact*n_nue0)

    axes[0].plot(t_array, nex_array[:,0], 'b-', label=r'${\rm Re}[N_{ex}]$')
    axes[0].plot(t_array, nex_array[:,1], 'r--', label=r'${\rm Im}[N_{ex}]$')
    axes[1].plot(t_array, nex_array[:,2], 'b-', label=r'${\rm Re}[\overline{N}_{ex}]$')
    axes[1].plot(t_array, nex_array[:,3], 'r--', label=r'${\rm Im}[\overline{N}_{ex}]$')

    axes[0].set_title(r'${{\rm Cell\,\,ID}} = {}$'.format(cell_id))


    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    plt.savefig("Nex_comps_vs_time_cell_{}.pdf".format(cell_id), bbox_inches="tight")

    fig.clf()
