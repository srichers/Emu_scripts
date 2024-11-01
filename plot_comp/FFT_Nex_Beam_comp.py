# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
import os

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
k_str = ["k", "k(1|cm)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]
N00_str = ["N00_FFT", "N00_FFT(cm^-2)"]
N11_str = ["N11_FFT", "N11_FFT(cm^-2)"]
N01_str = ["N01_FFT", "N01_FFT(cm^-2)"]

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename_FFT, filename_avg, tind, Nex_ind):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]

    fftData = h5py.File(filename_FFT,"r")
    t=np.array(fftData[t_str[tind]])
    k=np.array(fftData[k_str[tind]])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData[N00_str[tind]])
    Nxx=np.array(fftData[N11_str[tind]])
    Nex=np.array(fftData[N01_str[tind]])
    fftData.close()

    if Nex_ind == -1:
        # make time relative to tmax
        avgData = h5py.File(filename_avg,"r")
        Nexavg=np.array(avgData[N_str[tind]][:,0,1])
        avgData.close()
        itmax = np.argmax(Nexavg)
        it = int(itmax/2)
    else:
        it = Nex_ind

    print(it,t[it])
    trace = Nee[it,0]+Nxx[it,0]
    return k, (Nex/trace)[it, :-1]


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
#emu 1024 L=32.0 cm
#emu_base = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series_nx1024_Lz32.0/"
#emu_h5 = "plt_reduced_data.h5"
#emu_fft_h5 = "plt_reduced_data_fft_power.h5"
#emu_sims = ["neebar_{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
#emu_inds = -np.ones([num_sims], dtype=np.int64)
#emu_inds[0] = 16/2
#emu_inds[1] = 11/2
#emu_inds[2] = 9/2
#tind[0] = 1
#emu 1024 L=32.0 cm
emu_base = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series_nx1024_Lz8.0/"
emu_h5 = "plt_reduced_data.h5"
emu_fft_h5 = "plt_reduced_data_fft_power.h5"
emu_sims = ["neebar_{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
emu_inds = -np.ones([num_sims], dtype=np.int64)
emu_inds[0] = 69/2
emu_inds[1] = 47/2
emu_inds[2] = 38/2
emu_inds[3] = 33/2
emu_inds[4] = 29/2
tind[0] = 1

#MPC Beam/rand/changing_N_nuebar/*
flash_base = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/res_1024/"
flash_h5 = "reduced_data.h5"
flash_fft_h5 = "reduced_data_fft_power.h5"
flash_sims = ["sim{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
flash_inds = -np.ones([num_sims], dtype=np.int64)
flash_inds[0] = 82/2
flash_inds[1] = 59/2
flash_inds[4] = 37/2
kend = np.ones([num_sims], dtype=np.int64)
kend[0] = 9
kend[1] = 12
kend[2] = 14
kend[3] = 16
kend[4] = 18
savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/res_1024/comp_emu_1024_L8/FFT_Nex_Beam_comp.pdf"
tind[1] = 0

##nuM Beam/rand/changing_N_nuebar/*
#flash_base = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/nuM/changing_N_nuebar/res_1024/"
#flash_h5 = "reduced_data.h5"
#flash_fft_h5 = "reduced_data_fft_power.h5"
#flash_sims = ["sim{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
#flash_inds = -np.ones([num_sims], dtype=np.int64)
#flash_inds[1] = 57/2
#flash_inds[2] = 48/2
#flash_inds[3] = 40/2
#kend = np.ones([num_sims], dtype=np.int64)
#kend[0] = 10
#kend[1] = 11
#kend[2] = 13
#kend[3] = 16
#kend[4] = 18
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/nuM/changing_N_nuebar/res_1024/comp_emu_L8/FFT_Nex_Beam_comp.pdf"
#tind[1] = 0


fig, axes = plt.subplots(num_sims,1, figsize=(3*num_sims,5*num_sims), sharex=True)
plt.subplots_adjust(hspace=0)

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel(r"$\mathcal{D}(k)$", fontsize=48, labelpad=50)
plt.xlabel(r"$k\,({\rm cm}^{-1})$", fontsize=48)
axes[0].set_title(r"${\rm Beam\,\,Tests}$", fontsize=48)

ytick_vals = 10.0**(np.linspace(-14,-8, 4))
ylabels = [r'$10^{{{}}}$'.format(p) for p in range(-14,-7,2)]

for ax in axes:    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.set_xlim(-1.0, 40.0)
    ax.set_ylim(1.5e-16,1.e-7)
    ax.set_yticks(ytick_vals, labels=ylabels, fontsize=36)
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.tick_params(axis='x',  labelsize=36)



for i in range(num_sims):
    print("N_{ee}/\\bar{N}_{ee} = ", 0.2*(i+1))

    filename_emu = emu_base + emu_sims[i] + emu_h5
    filename_emu_fft = emu_base + emu_sims[i] + emu_fft_h5
    if i == 0:
        #Emu data has dimensionful values:
        avgData = h5py.File(filename_emu,"r")
        nbase=np.array(avgData[N_str[1]][0,0,0])
        avgData.close()
    k,N_ex = plotdata(filename_emu_fft,filename_emu,tind[0],emu_inds[i])
    axes[i].semilogy(k, N_ex, 'k--', zorder=1)
    print('Emu, kmax = ', k[np.argmax(N_ex)])

    filename_flash = flash_base + flash_sims[i] + flash_h5
    filename_flash_fft = flash_base + flash_sims[i] + flash_fft_h5
    k,N_ex = plotdata(filename_flash_fft,filename_flash,tind[1],flash_inds[i])
    axes[i].semilogy(k, N_ex, 'r-', zorder=2)
    print('FLASH, kmax = ', k[np.argmax(N_ex)])
    print('FLASH, k_end/2 = ', k[kend[i]]/2.0)

    mu = np.sqrt(2.0)*gfermi*nbase*(1.0 + float(0.2*(i+1)))/2.0*hbarc**3/hbar
    axes[i].axvline(x=2.0*mu/clight, color='g')

    #axes[i].axvline(x=k[kend[i]], color='m')

    fig.text(0.66, float(num_sims - i - 1)*0.775/float(num_sims) + 0.21, \
            r'$\alpha = {:.1f}$'.format(0.2*(i+1)), fontsize=36)

#ax.legend(loc='upper right', fontsize=20, frameon=False)
plt.savefig(savename, bbox_inches="tight")
