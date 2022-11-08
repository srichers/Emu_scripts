# Uses vertical axes of Q/N_ee(t=0)

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#NSM_1:
n_nue0_1 = 1.421954234999705e+33     # 1/ccm
n_nux0_1 = 1.9645407875568215e+33/4. # 1/ccm, each flavor
n_tot_1 = n_nue0_1 + 2.*n_nux0_1
n_2F_1 = n_nue0_1 + n_nux0_1
n_tot_eq_1 = n_tot_1/3.0
n_2F_eq_1 = n_2F_1/2.0

#NSM_2:
n_nue0_2 = 2.3293607911671233e+33    # 1/ccm
n_nux0_2 = 1.5026785300973756e+33 # 1/ccm, each flavor
n_tot_2 = n_nue0_2 + 2.*n_nux0_2
n_2F_2 = n_nue0_2 + n_nux0_2
n_tot_eq_2 = n_tot_2/3.0
n_2F_eq_2 = n_2F_2/2.0

#NSM_3:
n_nue0_3 = 2.8800567085107055e+33    # 1/ccm
n_nux0_3 = 4.831622183948198e+32 # 1/ccm, each flavor
n_tot_3 = n_nue0_3 + 2.*n_nux0_3
n_2F_3 = n_nue0_3 + n_nux0_3
n_tot_eq_3 = n_tot_3/3.0
n_2F_eq_3 = n_2F_3/2.0

n_tot = [n_tot_1, n_tot_2, n_tot_3]
n_2F = [n_2F_1, n_2F_2, n_2F_3]
n_2F_eq = [n_2F_eq_1, n_2F_eq_2, n_2F_eq_3]

mfact = 1.e-33
mfactstr = r'$10^{-33}\times$'

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

def plotdata_new_format(filename,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t(s)"])*1e9
    N=np.array(avgData["N_avg_mag(1|ccm)"])[:,a,b]
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


emu_2f_pre = "/global/project/projectdirs/m3761/FLASH/Emu/"

#emu_2f_sims = ['NSM_1/']
emu_2f_sims = ['NSM_1/', 'NSM_2/32dir/', 'NSM_3/32dir/']
emu_2f_res = ['merger_2F/', 'merger_2F_lowres/']


emu_3f_pre = "/global/project/projectdirs/m3761/FLASH/Emu/"

emu_3f_sims = ['NSM_1/']
emu_3f_res = ['merger_3F/', 'merger_3F_lowres/']


bang_pre = "/global/project/projectdirs/m3761/FLASH/FFI_3D/"

bang_sims = ['NSM_1/', 'NSM_2/t3/', 'NSM_3/t6/']
bang_res = ['sim/', 'res_a/', 'res_b/']


test_fig_labels = [r'${\rm NSM}\,1$', r'${\rm NSM}\,2$', r'${\rm NSM}\,3$']

h5_filename_old = "reduced_data_old.h5"
h5_filename_orig = "reduced_data.h5"
h5_filename_emu = "plt_reduced_data.h5"

aval = [1.0, 0.5, 0.25]

for i in range(3):

    if i == 0:
        ax = plt.subplot(2,3,i+1)
        ax_ex = plt.subplot(2,3,i+4)
    else:
        ax = plt.subplot(2,3,i+1, sharex=ax0, sharey=ax0)
        ax_ex = plt.subplot(2,3,i+4, sharex=ax_ex0, sharey=ax_ex0)


    #############
    # plot data #
    #############
    for j in range(3):
        #if i == 0 and j < 2:
        #    if j == 1:
        #        if j == 0:
        #            h5_filename = h5_filename_emu
        #        else:
        #            h5_filename = h5_filename_orig
        if j < 2:
            h5_filename = h5_filename_emu
            if i == 0 and j == 1:
                h5_filename = h5_filename_orig
            if i == 0 and j == 0:
                print("no permissions")
            else:
                filename_emu_2f = emu_2f_pre + emu_2f_sims[i] + emu_2f_res[j] + h5_filename
                if i == 0:
                    t,N = plotdata(filename_emu_2f,0,0)
                    t_ex,N_ex = plotdata(filename_emu_2f,0,1)
                else:
                    t,N = plotdata_new_format(filename_emu_2f,0,0)
                    t_ex,N_ex = plotdata_new_format(filename_emu_2f,0,1)
                tmax = t[np.argmax(N_ex)]
                ax.plot(t-tmax, N/N[0], 'k-', alpha=aval[j], label=None)
                if j == 0:
                    ax_ex.semilogy(t-tmax, N_ex/N[0], 'k-', alpha=aval[j], label=r'${\rm {\tt EMU}\,\,(2f)}$')
                else:
                    ax_ex.semilogy(t-tmax, N_ex/N[0], 'k-', alpha=aval[j], label=None)

        if i == 0 and j < 2:
            if j == 0:
                h5_filename = h5_filename_orig
            else:
                h5_filename = h5_filename_orig
            filename_emu_3f = emu_3f_pre + emu_3f_sims[i] + emu_3f_res[j] + h5_filename
            t,N = plotdata(filename_emu_3f,0,0)
            t_ex,N_ex = plotdata(filename_emu_3f,0,1)
            tmax = t[np.argmax(N_ex)]
            ax.plot(t-tmax, N/N[0], 'k--', alpha=aval[j], label=None)
            if j == 0:
                ax_ex.semilogy(t-tmax, N_ex/N[0], 'k--', alpha=aval[j], label=r'${\rm {\tt EMU}\,\,(3f)}$')
            else:
                ax_ex.semilogy(t-tmax, N_ex/N[0], 'k--', alpha=aval[j], label=None)

        h5_filename = h5_filename_orig
        #special cases:
        if i == 0 and j == 2:
            filename_bang = bang_pre + bang_sims[i] + 'res_c/' + h5_filename
        else:
            filename_bang = bang_pre + bang_sims[i] + bang_res[j] + h5_filename
        t,N = plotdata(filename_bang,0,0)
        if j == 0:
            #special cases:
            if i == 0:
                N_fe = n_2F_eq[i]/(N[0]*n_2F[i])
            else:
                N_fe = n_2F_eq[i]/(N[0]*n_2F[i])
        t_ex,N_ex = plotdata(filename_bang,0,1)
        tmax = t[np.argmax(N_ex)]
        #special cases:
        if i == 0 and j == 1:
            ax.plot(t-tmax, N/N[0], 'r-', alpha=aval[j],  label=None)
        elif i == 0 and j == 2:
            ax.plot(t-tmax, N/N[0], 'r--', alpha=aval[j],  label=None)
        else:
            ax.plot(t-tmax, N/N[0], 'r-', alpha=aval[j],  label=None)
        ax.text(x=1.5, y=0.9, s=test_fig_labels[i], fontsize=12)
        if j == 0:
            ax_ex.semilogy(t-tmax, N_ex/N[0], 'r-', alpha=aval[j], label=r'${\rm {\tt FLASH}\,\,(2f)}$')
        elif i == 0 and j == 2:
            ax_ex.semilogy(t-tmax, N_ex/N[0], 'r--', alpha=aval[j], label=None)
        else:
            ax_ex.semilogy(t-tmax, N_ex/N[0], 'r-', alpha=aval[j], label=None)

    ax_ex.set_xlabel(r"$t-t_{\rm max}\,(10^{-9}\,{\rm s})$")
    plt.setp(ax.get_xticklabels(), visible=False)

    if i == 0:
        ax.set_ylabel(r"$\langle N_{ee}(t)/N_{ee}(0)\rangle$")
        ax_ex.set_ylabel(r"$\langle |N_{ex}(t)|/N_{ee}(0)\rangle$")
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
    ax.axhline(N_fe, color="green")
    #ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.set_xlim([-0.5,2.0])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

    ax_ex.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax_ex.set_xlim([-0.5,2.0])
    ax_ex.xaxis.set_minor_locator(AutoMinorLocator())
    ax_ex.yaxis.set_minor_locator(AutoMinorLocator())
    ax_ex.yaxis.set_major_locator(LogLocator(numticks=5))
    ax_ex.minorticks_on()

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("N_N0_comp_3NSM_ee_ex.pdf", bbox_inches="tight")
