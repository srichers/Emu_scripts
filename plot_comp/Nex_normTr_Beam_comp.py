# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)


gfermi = 1.1663787e-11 #MeV^{-2}
hbarc = 1.97326966e-11 #MeV cm
hbar = 6.582119569e-22 #MeV s
clight = 29979245800.0 #cm/s

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
def plotdata(filename,a,b,tind):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData[t_str[tind]])*1e9
    N=np.array(avgData[N_str[tind]])[:,a,b]
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


fig, ax = plt.subplots(1,1, figsize=(6,5), sharex=True)

#############
# plot data #
#############

num_sims = 5
tind = np.ones([2], dtype=np.int64)

##emu 256
#emu_base = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series/"
#emu_h5 = "reduced0D_selection.h5"
#emu_sims = ["neebar_{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
#emu_inds = -np.ones([num_sims], dtype=np.int)
#emu_inds[0] = 402
#emu_inds[1] = 287
#emu_inds[2] = 237
#emu_inds[4] = 177
##emu 1024
#emu_base = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series_nx1024/"
#emu_h5 = "plt_reduced_data.h5"
#emu_sims = ["neebar_{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
#emu_inds = -np.ones([num_sims], dtype=np.int32)
#emu_inds[0] = 16
#emu_inds[1] = 11
#emu_inds[2] = 9
#tind[0] = 1
#t_emu_off = np.zeros([num_sims])
#t_emu_off[0] = 0.005
#t_emu_off[3] = 0.002
emu_base = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series_nx1024_Lz8.0/"
emu_h5 = "plt_reduced_data.h5"
emu_sims = ["neebar_{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
emu_inds = -np.ones([num_sims], dtype=np.int32)
emu_inds[0] = 69
emu_inds[1] = 47
emu_inds[2] = 38
emu_inds[3] = 33
emu_inds[4] = 29
tind[0] = 1
t_emu_off = np.zeros([num_sims])
t_emu_off[0] = 0.005
t_emu_off[1] = -0.0005
t_emu_off[3] = 0.002

#Beam/rand/changing_N_nuebar/*
flash_base = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/res_1024/"
flash_h5 = "reduced_data.h5"
flash_sims = ["sim{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
flash_inds = -np.ones([num_sims], dtype=np.int32)
flash_inds[0] = 82
flash_inds[1] = 59
flash_inds[4] = 37
savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/res_1024/comp_emu_1024_L8/Nex_normTr_Beam_comp.pdf"
tind[1] = 0
xlim = (-0.15, 0.01)


for i in range(num_sims):
    print("N_{ee}/\\bar{N}_{ee} = ", 0.2*(i+1))

    filename_emu = emu_base + emu_sims[i] + emu_h5
    t,Nee = plotdata(filename_emu,0,0,tind[0])
    t,Nxx = plotdata(filename_emu,1,1,tind[0])
    t_ex,N_ex = plotdata(filename_emu,0,1,tind[0])
    #normalize emu data:
    Ntr_2f = Nee[0] + Nxx[0]
    N_ex = N_ex/Ntr_2f
    if emu_inds[i] != -1:
        tmax_ind = emu_inds[i]
    else:
        tmax_ind = np.argmax(N_ex)
    tmax = t[tmax_ind]
    if i == num_sims-1:
        #ax.semilogy(t-tmax, N_ex, 'k--', alpha=0.2*(i+1), zorder=1, label=r'${\rm {\tt Emu}}$')
        ax.semilogy(t-tmax+t_emu_off[i], N_ex/N_ex[tmax_ind], 'k--', alpha=0.2*(i+1), zorder=1, label=r'${\rm {\tt Emu}}$')
    else:
        #ax.semilogy(t-tmax, N_ex, 'k--', alpha=0.2*(i+1), zorder=1)
        ax.semilogy(t-tmax+t_emu_off[i], N_ex/N_ex[tmax_ind], 'k--', alpha=0.2*(i+1), zorder=1)
    #a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
    #print('Emu (2F)')
    #print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])
    print('Emu, tmax_ind = ', tmax_ind)
    #Emu has dimensionful values; use for calculating expected growth rates:
    nbase = Nee[0]

    filename_flash = flash_base + flash_sims[i] + flash_h5
    t,N_ex = plotdata(filename_flash,0,1,tind[1])
    if flash_inds[i] != -1:
        tmax_ind = flash_inds[i]
    else:
        tmax_ind = np.argmax(N_ex)
    tmax = t[tmax_ind]
    if i == num_sims-1:
        #ax.semilogy(t-tmax, N_ex, 'r-', alpha=0.2*(i+1), zorder=1, label=r'${\rm {\tt FLASH}}$')
        ax.semilogy(t-tmax, N_ex/N_ex[tmax_ind], 'r-', alpha=0.2*(i+1), zorder=1, label=r'${\rm {\tt FLASH}}$')
    else:
        #ax.semilogy(t-tmax, N_ex, 'r-', alpha=0.2*(i+1), zorder=1)
        ax.semilogy(t-tmax, N_ex/N_ex[tmax_ind], 'r-', alpha=0.2*(i+1), zorder=1)
    #a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
    #print('FLASH')
    #print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])
    print('FLASH, tmax_ind = ', tmax_ind)

    mu = np.sqrt(2.0)*gfermi*nbase*(1.0 + float(0.2*(i+1)))/2.0*hbarc**3/hbar
    asymm = (1.0 - float(0.2*(i+1)))/(1.0 + float(0.2*(i+1)))
    growth_rate = 2.0*mu*np.sqrt(1.0 - asymm**2)
    print("Analytic growth rate: ", growth_rate/1.e10, " e10 s^{-1}")
    print("Analytic kmax: ", 2.0*mu/clight, " cm^{-1}")

##############
# formatting #
##############
ax.set_xlabel(r"$t-t_{\rm off}\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
if "xlim" in locals():
    ax.set_xlim(xlim[0], xlim[1])
ax.set_ylabel(r"$\langle |N_{ex}|\rangle\,\,({\rm a.u.})$")
ax.set_title(r"${\rm Beam\,\,Tests}$")

ax.legend(loc='lower right', fontsize=20, frameon=False)
plt.savefig(savename, bbox_inches="tight")
