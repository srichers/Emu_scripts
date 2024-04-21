# ffmpeg -framerate 10 -pattern_type glob -i '*.png' video.mp4

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

variables=["N","Fx","Fy","Fz"]
flavors=["00","11","22","01","02","12"]
cmap=mpl.cm.jet

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
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


def makeplot(v,f,data1, data2):
    plt.close('all')

    # get appropriate data
    t1=data1["t"][::2]
    t2=data2["t"][::5]
    k1=data1["k"]
    k2=data2["k"]
    fft1 = data1[v+f+"_FFT"][::2,:]
    fft2 = data2[v+f+"_FFT"][::5,:]
    total_power1 = np.max(np.sum(fft1,axis=1))
    total_power2 = np.max(np.sum(fft2,axis=1))
    ymax1 = 1.1*np.max(fft1[:,:-1]/total_power1)
    ymin1 = np.min(fft1[:,:-1]/total_power1)
    ymax2 = 1.1*np.max(fft2[:,:-1]/total_power2)
    ymin2 = np.min(fft2[:,:-1]/total_power2)
    ymax = max(ymax1,ymax2)
    ymin = min(ymin1,ymin2)
    nt = min(len(t1),len(t2))
    for it in range(nt):
        print(it)
        plt.clf()
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.semilogy(k1, fft1[it,:-1]/total_power1, color=cmap(t1[it]/t1[nt-1]))
        ax.semilogy(k2, fft2[it,:-1]/total_power2, color=cmap(t2[it]/t2[nt-1]))
    
        # colorbar
        cax = fig.add_axes([0.125, .89, .775, 0.02])
        cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
        norm = mpl.colors.Normalize(vmin=0, vmax=t1[nt-1]*1e9)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,orientation='horizontal')
        cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
        cax.minorticks_on()
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cax.xaxis.set_minor_locator(MultipleLocator(0.1))

        # axis labels
        ax.set_xlabel(r"$k\,(\mathrm{cm}^{-1})$")
        ax.set_ylabel(r"$|\widetilde{f}|^2\,(\mathrm{cm}^{-2})$")
        ax.set_ylim([ymin,ymax])
        ax.text(k1[0],.01,("t1=%.2e"%t1[it])+("   t2=%.2e"%t2[it]))
        
        plt.savefig(v+f+"_FFT_power_"+str(it).zfill(5)+".png", bbox_inches='tight')

data1 = h5py.File("global/project/projectdirs/m3761/3D/128r64d_128n800s16mpi/reduced_data_fft_power.h5","r")
data2 = h5py.File("ocean/projects/phy200048p/shared/3D/fiducial_3D/1/reduced_data_fft_power.h5","r")

v="N"
f="01"
makeplot(v,f, data1, data2)

data1.close()
data2.close()
