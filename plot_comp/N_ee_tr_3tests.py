# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)


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
    N=np.array(avgData["N_avg_mag"])[:,a,b]
    avgData.close()
    return t, N

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
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


fig = plt.figure(figsize=(12,3))

test_list = [['Fiducial', 'fid'], ['90Degree', '90d'], ['TwoThirds', '2_3']]

emu_2f_pre = "/global/project/projectdirs/m3761/Evan/"
emu_2f_suf = "_3D_2F/reduced_data.h5"

emu_3f_pre = "/global/project/projectdirs/m3761/Evan/"
emu_3f_suf = "_3D_3F_reduced_data.h5"

bang_pre = "/global/project/projectdirs/m3761/FLASH/FFI_3D/"
bang_suf = "/sim1/reduced_data_nov4_test_hdf5_chk.h5"

for i,test in enumerate(test_list):

    if i == 0:
        ax = plt.subplot(1,3,i+1)
    else:
        ax = plt.subplot(1,3,i+1, sharex=ax0, sharey=ax0)


    ##############
    # formatting #
    ##############
    ax.axhline(1./2., color="green")
    #ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.set_xlim([0.0,5.0])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    #ax.grid(which='both')
    
    #############
    # plot data #
    #############
    filename_emu_2f = emu_2f_pre + test[0] + emu_2f_suf
    t,N = plotdata(filename_emu_2f,0,0)
    ax.plot(t, N, 'k-', label=r'${\rm Emu\,\,(2f)}$')

    filename_emu_3f = emu_3f_pre + test[0] + emu_3f_suf
    t,N = plotdata(filename_emu_3f,0,0)
    #special code for excising a single point
    if test[0] == test_list[0][0]:
        bad_ind = 152
        t = np.concatenate((t[:bad_ind-1], t[bad_ind+1:]))
        N = np.concatenate((N[:bad_ind-1], N[bad_ind+1:]))
    ax.plot(t, N, 'k--', label=r'${\rm Emu\,\,(3f)}$')

    filename_bang = bang_pre + test[1] + bang_suf
    t,N = plotdata(filename_bang,0,0)
    ax.plot(t, N, 'r-', label=r'${\rm FLASH\,\,(2f)}$')
    ax.set_xlabel(r"$t\,(10^{-9}\,{\rm s})$")
    if i == 0:
        ax.set_ylabel(r"$\langle N_{ee}/{\rm Tr}[N]\rangle$")
        ax.legend(loc='upper right', fontsize=12, frameon=False)
        ax0 = ax
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

plt.subplots_adjust(wspace=0)
plt.savefig("N_ee_tr_comp_3tests.pdf", bbox_inches="tight")
