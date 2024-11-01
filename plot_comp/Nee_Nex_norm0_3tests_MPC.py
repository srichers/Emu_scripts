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
    
t_str = ["t", "t(s)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename,a,b,ind):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData[t_str[ind]])*1e9
    N=np.array(avgData[N_str[ind]])[:,a,b]
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

test_list = [['Fiducial', 'fid/MPC/d_pert/t5/'], ['90Degree', '90d/MPC/d_pert/t3/'], ['TwoThirds', '2_3/MPC/d_pert/t3/']]
test_fig_labels = [r'${\rm Fiducial}$', r'${\rm 90Degree}$', r'${\rm TwoThirds}$']

est_imo = np.empty([3])
Nex_pow = np.empty([3])
scale_fact = np.empty([3])
ind_offset = np.empty([3], dtype=np.int8)
ind1_array = np.empty([3], dtype=np.int8)
#fid from LSA:
est_imo[0] = 7.04e10 #s^{-1}
Nex_pow[0] = -5.0
scale_fact[0] = 450.0
ind_offset[0] = 2
ind1_array[0] = -1
#90d from LSA:
est_imo[1] = 5.01e10 #s^{-1}
Nex_pow[1] = -4.5
scale_fact[1] = 500.0
ind_offset[1] = 10
ind1_array[1] = -1
#2/3 from LSA:
est_imo[2] = 1.57e10 #s^{-1}
Nex_pow[2] = -3.5
scale_fact[2] = 12.0
ind_offset[2] = 10
ind1_array[2] = -1


emu_2f_pre = "/global/cfs/projectdirs/m3761/FLASH/Emu/"
emu_2f_suf = "_3D_2F/reduced_data.h5"

#emu_3f_pre = "/global/cfs/projectdirs/m3761/FLASH/Emu/"
#emu_3f_suf = "_3D_3F/reduced_data.h5"

bang_pre = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/"
bang_suf = "xy_large/sim/reduced_data.h5"
flash_inds = [-1, 118, -1]

savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/3tests/MPC/emu_comp/"

ind = np.zeros([3,3], dtype=np.int8)
#special cases
ind[0,2] = 1

test_titles = [r'${\rm Fiducial}$', r'${\rm 90Degree}$', r'${\rm TwoThirds}$']

delta_max_dec = 0.2

for i,test in enumerate(test_list):

    if i == 0:
        ax = plt.subplot(2,3,i+1)
        #ax_ex = plt.subplot(2,3,i+4)
        ax_ex = plt.subplot(2,3,i+4, sharex=ax)
    else:
        #ax = plt.subplot(2,3,i+1, sharex=ax0, sharey=ax0)
        #ax_ex = plt.subplot(2,3,i+4, sharex=ax_ex0, sharey=ax_ex0)
        ax = plt.subplot(2,3,i+1, sharey=ax0)
        ax_ex = plt.subplot(2,3,i+4, sharex=ax, sharey=ax_ex0)

    print(test_titles[i])


    #############
    # plot data #
    #############
    filename_emu_2f = emu_2f_pre + test[0] + emu_2f_suf
    t,N = plotdata(filename_emu_2f,0,0,ind[0,i])
    t_ex,N_ex = plotdata(filename_emu_2f,0,1,ind[0,i])
    #special code for renormalizing emu_2F:
    N0 = N[0]
    N = N/N0
    N_ex = N_ex/N0
    tmax = t[np.argmax(N_ex)]
    tdec = tmax + delta_max_dec #decoherence time after sat.
    tdec_ind = np.argmin(abs(t-tdec)) #index where tdec falls in t
    ax.plot(t-tmax, N, 'k--', label=None)
    ax_ex.semilogy(t-tmax, N_ex, 'k--', label=r'${\rm {\tt Emu}}$')
    print('Emu (2f)')
    print('N_ex max = ', N_ex[np.argmax(N_ex)])
    print('N_ex dec = ', N_ex[tdec_ind])
    print('N_ee asymp = ', N[-1])


    #filename_emu_3f = emu_3f_pre + test[0] + emu_3f_suf
    #t,N = plotdata(filename_emu_3f,0,0,ind[1,i])
    #t_ex,N_ex = plotdata(filename_emu_3f,0,1,ind[1,i])
    #tmax = t[np.argmax(N_ex)]
    ##special code for excising a single point
    #if test[0] == test_list[0][0]:
    #    bad_ind = 152
    #    t = np.concatenate((t[:bad_ind-1], t[bad_ind+1:]))
    #    N = np.concatenate((N[:bad_ind-1], N[bad_ind+1:]))
    #    t_ex = np.concatenate((t_ex[:bad_ind-1], t_ex[bad_ind+1:]))
    #    N_ex = np.concatenate((N_ex[:bad_ind-1], N_ex[bad_ind+1:]))
    #ax.plot(t-tmax, N, 'k--', label=None)
    #ax_ex.semilogy(t-tmax, N_ex, 'k--', label=r'${\rm {\tt EMU}\,\,(3f)}$')

    filename_bang = bang_pre + test[1] + bang_suf
    t,N = plotdata(filename_bang,0,0,ind[2,i])
    t_ex,N_ex = plotdata(filename_bang,0,1,ind[2,i])
    if flash_inds[i] != -1:
        tmax_ind = flash_inds[i]
    else:
        tmax_ind = np.argmax(N_ex)
    tmax = t[tmax_ind]
    tdec = tmax + delta_max_dec #decoherence time after sat.
    tdec_ind = np.argmin(abs(t-tdec)) #index where tdec falls in t
    ax.plot(t-tmax, N, 'r-', label=None, zorder=0)
    #ax.text(x=2.5, y=0.9, s=test_fig_labels[i], fontsize=12)
    ax_ex.set_xlabel(r"$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$")
    ax_ex.semilogy(t-tmax, N_ex, 'r-', label=r'${\rm {\tt FLASH}}$', zorder=0)
    print('Flash')
    print('N_ex max = ', N_ex[np.argmax(N_ex)])
    print('N_ex dec = ', N_ex[tdec_ind])
    print('N_ee asymp = ', N[-1])

    #plot LSA growth rate
    est_imo_ns = est_imo[i]/1.e+9 #ns^{-1}
    Nex_base = 10.0**(Nex_pow[i])
    if ind1_array[i] == -1:
        ind1 = np.argmin(np.abs(np.log(N_ex[1:]/Nex_base)))
    else:
        ind1 = ind1_array[i]
    ind2 = ind1 + ind_offset[i]
    t_line = [t[ind1], t[ind2]]
    N_line = [scale_fact[i]*N_ex[ind1], scale_fact[i]*N_ex[ind1]*np.exp(est_imo_ns*(t[ind2] - t[ind1]))]
    ax_ex.semilogy(t_line-tmax, N_line, color='orange', label=None)

    plt.setp(ax.get_xticklabels(), visible=False)
    if i == 0:
        ax.set_ylabel(r"$\langle N_{ee}\rangle/\langle{\rm Tr}[N]\rangle$")
        ax_ex.set_ylabel(r"$\langle |N_{ex}|\rangle/\langle{\rm Tr}[N]\rangle$")
        #ax.set_ylabel(r"$\langle N_{ee}(t)/N_{ee}(0)\rangle$")
        #ax_ex.set_ylabel(r"$\langle |N_{ex}(t)|/N_{ee}(0)\rangle$")
        #ytick_vals = [1.e-7, 1.e-5, 1.e-3, 1.e-1]
        #ytick_labs = [r'$10^{{{}}}$'.format(num) for num in [-7, -5, -3, -1]]
        #ax_ex.set_yticks(ytick_vals)
        #ax_ex.set_yticklabels(ytick_labs)
        ax_ex.legend(loc='lower right', fontsize=20, frameon=False)
        ax0 = ax
        ax_ex0 = ax_ex
    else:
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax_ex.get_yticklabels(), visible=False)

    #if i == 2:
    #    ax_ex.legend(loc='lower right', fontsize=12, frameon=False)

    if i ==2:
        ax.set_xlim([-1.9,4.0])
    else:
        ax.set_xlim([-1.0,4.0])

    ##############
    # formatting #
    ##############
    ax.axhline(1./2., color="green",zorder=0)
    #ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.set_title(test_titles[i])

    ax_ex.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax_ex.xaxis.set_minor_locator(AutoMinorLocator())
    ax_ex.yaxis.set_minor_locator(AutoMinorLocator())
    ax_ex.yaxis.set_major_locator(LogLocator(numticks=5))
    ax_ex.minorticks_on()

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(savedir + "Nee_Nex_norm0_3tests_MPC.pdf", bbox_inches="tight")
