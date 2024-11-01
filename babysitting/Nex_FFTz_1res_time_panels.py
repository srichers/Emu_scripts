import os
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


#############
# plot data #
#############

basename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_4/d_pert/t2/xy_large/sim/"
filename_FFT   = basename + "reduced_data_fftz_power.h5"
filename_avg   = basename + "reduced_data.h5"

avgData = h5py.File(filename_avg,"r")
t=np.array(avgData["t"])*1e9
tsat = t[np.argmax(np.array(avgData["N_avg_mag"])[:,0,1])]
t = t - tsat
avgData.close()

plot_times = [t[1], -1.0, t[72], -0.1, 0.0, 2.0] #wrt saturation

fftData = h5py.File(filename_FFT,"r")
k=np.array(fftData["k"])
#convert from 1/\lambda to k=2\pi/\lambda:
k = 2.0*np.pi*k

kmax = 2.05 #cm^{-1}

Nee=np.array(fftData["N00_FFT"])
Nxx=np.array(fftData["N11_FFT"])
Nex=np.array(fftData["N01_FFT"])

trace = Nee[0,np.argmin(np.abs(k))]+Nxx[0,np.argmin(np.abs(k))]
Nex = Nex/trace

fftData.close()

fig, axes = plt.subplots(len(plot_times),1, figsize=(3*len(plot_times),5*len(plot_times)), sharex=True)
plt.subplots_adjust(hspace=0)

# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel(r"$\mathcal{D}(k_{\rm z}\,;t)$", fontsize=48, labelpad=50)
plt.xlabel(r"$k_{\rm z}\,({\rm cm}^{-1})$", fontsize=48)
axes[0].set_title(r"${\rm NSM}4$", fontsize=48)


num_ticks = [5, 5, 5, 5, 5, 5]
floor_array = [-24, -18, -10, -9, -8, -10]
ceil_array =  [-8,  -6,  -6,  -5, -4, -6]

for i in range(0,len(plot_times)):

    ax = axes[i]

    ind = np.argmin(abs(t-plot_times[i]))
    N3 = Nex[ind, :]
    ind3 = np.argmax(N3)
    ax.semilogy(k, N3, 'r-')
    ax.axvline(0.0, color="k", linestyle="--")
    ax.axvline(kmax, color='g', label=None)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(-34.0, 34.0)
    ax.tick_params(axis='x', which='both', direction='in', top=True, labelsize=36)
    floor = floor_array[i]
    ceil = ceil_array[i]
    ytick_vals = 10.0**(np.linspace(floor,ceil, num_ticks[i]))
    ylabels = [r"$10^{{{}}}$".format(int(exp)) for exp in np.log10(ytick_vals)]
    ax.set_yticks(ytick_vals, labels=ylabels, fontsize=36)
    ax.minorticks_off()
    ax.tick_params(axis='y', which='major', direction='in', right=True)
    ax.tick_params(axis='y', which='minor', direction='in', left=False)
    time = t[ind]
    if time != 0.0:
        time_power = int(np.floor(np.log10(abs(time))))
    else:
        time_power = 0
    if time_power < 0:
        fig.text(0.16, float(len(plot_times) - i - 1)*0.775/float(len(plot_times)) + 0.21, \
                r'$t-t_{{\rm sat}}={:.2}{{\rm e}}\textnormal{{--}}{}\,{{\rm ns}}$'.format(10.0**(-time_power)*float(time), abs(time_power)), fontsize=36)
    elif time_power > 0:
        fig.text(0.16, float(len(plot_times) - i - 1)*0.775/float(len(plot_times)) + 0.21, \
                r'$t-t_{{\rm sat}}={:.2}{{\rm e}}\textnormal{{+}}{}\,{{\rm ns}}$'.format(10.0**(-time_power)*float(time), abs(time_power)), fontsize=36)
    else:
        fig.text(0.16, float(len(plot_times) - i - 1)*0.775/float(len(plot_times)) + 0.21, r'$t-t_{{\rm sat}}={:.2}\,{{\rm ns}}$'.format(float(time)), fontsize=36)


plt.savefig(basename + "Nex_FFTz_1res_time_panels.pdf", bbox_inches="tight")
