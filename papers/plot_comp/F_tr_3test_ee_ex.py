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
clight = 2.99e10

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)

emu_direction = 0
flash_direction = 1

######################
# read averaged data #
######################
def plotdata(filename,a,b, direction):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])[1:]*1e9
    N=np.array(avgData["F_avg_mag"])[1:,direction,a,b]
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


fig = plt.figure(figsize=(12,6))

test_list = [['Fiducial', 'fid'], ['90Degree', '90d'], ['TwoThirds', '2_3']]
test_fig_labels = [r'${\rm Fiducial}$', r'${\rm 90Degree}$', r'${\rm TwoThirds}$']

emu_2f_pre = "/global/project/projectdirs/m3761/Evan/"
emu_2f_suf = "_3D_2F/reduced_data.h5"

emu_3f_pre = "/global/project/projectdirs/m3761/Evan/"
emu_3f_suf = "_3D_3F_reduced_data.h5"

for i,test in enumerate(test_list):

    if i == 0:
        ax = plt.subplot(2,3,i+1)
        ax_ex = plt.subplot(2,3,i+4)
    else:
        ax = plt.subplot(2,3,i+1, sharex=ax0, sharey=ax0)
        ax_ex = plt.subplot(2,3,i+4, sharex=ax_ex0, sharey=ax_ex0)


    #############
    # plot data #
    #############
    filename_emu_2f = emu_2f_pre + test[0] + emu_2f_suf
    t,N = plotdata(filename_emu_2f,0,0,emu_direction)
    t_ex,N_ex = plotdata(filename_emu_2f,0,1, emu_direction)
    tmax = 0 #t[np.argmax(N_ex)]
    ax.semilogy(t-tmax, N, 'k-', label=None)
    ax_ex.semilogy(t-tmax, N_ex, 'k-', label=r'${\rm Emu\,\,(2f)}$')

    filename_emu_3f = emu_3f_pre + test[0] + "_3D_3F_reduced_data.h5"
    t,N = plotdata(filename_emu_3f,0,0, emu_direction)
    t_ex,N_ex = plotdata(filename_emu_3f,0,1, emu_direction)
    tmax = 0 #t[np.argmax(N_ex)]
    #special code for excising a single point
    if test[0] == test_list[0][0]:
        bad_ind = 152
        t = np.concatenate((t[:bad_ind-1], t[bad_ind+1:]))
        N = np.concatenate((N[:bad_ind-1], N[bad_ind+1:]))
        t_ex = np.concatenate((t_ex[:bad_ind-1], t_ex[bad_ind+1:]))
        N_ex = np.concatenate((N_ex[:bad_ind-1], N_ex[bad_ind+1:]))
    ax.semilogy(t-tmax, N, 'k--', label=None)
    ax_ex.semilogy(t-tmax, N_ex, 'k--', label=r'${\rm Emu\,\,(3f)}$')

    filename = emu_3f_pre + test[0] + "_3D_3F_32d_reduced_data.h5"
    t,N = plotdata(filename,0,0, emu_direction)
    t_ex,N_ex = plotdata(filename,0,1, emu_direction)
    tmax = 0 #t[np.argmax(N_ex)]
    ax.semilogy(t-tmax, N, 'k--', alpha=0.25, label=None)
    ax_ex.semilogy(t-tmax, N_ex, 'k--', alpha=0.25)

    filename_bang = "/global/project/projectdirs/m3761/FLASH/FFI_3D/" + test[1] + "/sim1/reduced_data_nov4_test_hdf5_chk.h5"
    t,N = plotdata(filename_bang,0,0, flash_direction)
    t_ex,N_ex = plotdata(filename_bang,0,1, flash_direction)
    tmax = 0 #t[np.argmax(N_ex)]
    ax.semilogy(t-tmax, N/clight, 'r-', label=None)
    ax.text(x=2.5, y=0.9, s=test_fig_labels[i], fontsize=12)
    ax_ex.set_xlabel(r"$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$")
    ax_ex.semilogy(t-tmax, N_ex/clight, 'r-', label=r'${\rm FLASH\,\,(2f)}$')

    for itest in range(1,3):
        filename_bang = "/global/project/projectdirs/m3761/FLASH/FFI_3D/" + test[1] + "/res_test"+str(itest)+"/reduced_data_nov4_test_hdf5_chk.h5"
        t,N = plotdata(filename_bang,0,0, flash_direction)
        t_ex,N_ex = plotdata(filename_bang,0,1, flash_direction)
        tmax = 0 #t[np.argmax(N_ex)]
        ax.semilogy(t-tmax, N/clight, 'r-', alpha=0.25)
        ax_ex.semilogy(t-tmax, N_ex/clight, 'r-', alpha=0.25)



    plt.setp(ax.get_xticklabels(), visible=False)
    if i == 0:
        ax.set_ylabel(r"$\langle N_{ee}/{\rm Tr}[N]\rangle$")
        ax_ex.set_ylabel(r"$\langle |N_{ex}|/{\rm Tr}[N]\rangle$")
        #ytick_vals = [1.e-7, 1.e-5, 1.e-3, 1.e-1]
        #ytick_labs = [r'$10^{{{}}}$'.format(num) for num in [-7, -5, -3, -1]]
        #ax_ex.set_yticks(ytick_vals)
        #ax_ex.set_yticklabels(ytick_labs)
        ax_ex.legend(loc='lower right', fontsize=12, frameon=False)
        ax0 = ax
        ax_ex0 = ax_ex
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax_ex.get_yticklabels(), visible=False)

    ##############
    # formatting #
    ##############
    ax.axhline(1./2., color="green")
    #ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.set_xlim([-1.0,4.0])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

    ax_ex.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax_ex.set_xlim([-1.0,4.0])
    ax_ex.xaxis.set_minor_locator(AutoMinorLocator())
    ax_ex.yaxis.set_minor_locator(AutoMinorLocator())
    ax_ex.yaxis.set_major_locator(LogLocator(numticks=5))
    ax_ex.minorticks_on()

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("N_tr_comp_3tests_ee_ex.pdf", bbox_inches="tight")
