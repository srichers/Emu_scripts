# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
import argparse

parser = argparse.ArgumentParser(description='Reads reduced_data_*.h5 file and creates plot')
parser.add_argument('-i', '--input', dest='inf', type=str, help='input hdf5 file', metavar='', default='reduced_data_0_0_0_0.h5')

args = parser.parse_args()

input_file = args.inf
print('Using ' + input_file + ' for input hdf5 file\n')
ifile_str = input_file[:-3]

ifile_args = ifile_str.split('_')
blk = ifile_args[-5]
print('Block: ' + blk + '\n')
ind = ifile_args[-3:]
print('Cell Indices: {} {} {}'.format(ind[0], ind[1], ind[2]) + '\n')


base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]
# real/imag
R=0
I=1
    

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_real"])[:,a,b]
    avgData.close()
    return t, N

def plotdata_imag(filename,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_imag"])[:,a,b]
    avgData.close()
    return t, N

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
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


fig, axes = plt.subplots(2,1, figsize=(6,10), sharex=True)
plt.subplots_adjust(hspace=0)
#plt.subplots_adjust(vspace=1)

##############
# formatting #
##############
axes[1].set_xlabel(r'$t\,(10^{-9}\,{\rm s})$')
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
axes[0].set_xlim(0.2,0.4)
axes[0].set_ylabel(r'$ N_{ii}/{\rm Tr}[N]$')
axes[1].set_ylabel(r'${\rm Co}[N_{ex}]/{\rm Tr}[N]$')
#axes[0].set_ylim(0.9*mfact*n_nux0, 1.1*mfact*n_nue0)

#############
# plot data #
#############
#fid
box_length = 8.0
n_grid = 128
#beam
#sim1:
#box_length = 1.0
#n_grid = 128
#sim2:
#box_length = 8.0
#n_grid = 1024
#2_3
#sim:
#box_length = 32.0
#n_grid = 128
#testing1:
#box_length = 8.0
#n_grid = 512
#NSM_1
#box_length = 7.865243034321406
#n_grid = 128
#NSM_3
#t1
#t2:
#sim:
#box_length = 4.132703957221158
#n_grid = 256
#res_a
#box_length = 2.066351978610579
#n_grid = 128
#t3:
#res_b:
#box_length = 8.265407914442315
#n_grid = 128
#t4:
#sim:
#box_length = 16.53081582888463
#n_grid = 512
#filename_bang = "reduced_data_NSM_res2.h5"

t,Nee = plotdata(input_file,0,0)
t,Nxx = plotdata(input_file,1,1)
t,Nr = plotdata(input_file,0,1)
t,Ni = plotdata_imag(input_file,0,1)
axes[0].plot(t, Nee, 'r-', label=r'$ee$')
axes[0].plot(t, Nxx, 'b-', label=r'$xx$')
#axes[0].axhline(mfact*n_2F_eq, color="green")

axes[1].plot(t, Nr, 'g-', label=r'${\rm Re}$')
axes[1].plot(t, Ni, 'k-', label=r'${\rm Im}$')

#fig.text(0.5, 0.82, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
#fig.text(0.5, 0.77, r'$N_{{gp}}={}^3$'.format(n_grid))
#fig.text(0.5, 0.77, r'$N_{{gp}}={}$'.format(n_grid))


axes[0].legend(loc='best', frameon=False)
axes[1].legend(loc='best', frameon=False)
plt.savefig("N_comp_vs_t_{}_{}_{}_{}.pdf".format(blk, ind[0], ind[1], ind[2]), bbox_inches="tight")
