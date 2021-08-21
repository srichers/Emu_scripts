import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#cmap=mpl.cm.jet
cmap=mpl.cm.tab20

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

tmax = 5.0e-9


def makeplot(ax,data):

    # get appropriate data
    t=data["t"]
    k=data["k"]
    for it in range(1,len(t)):
        ax.semilogy(k, data["N00_FFT"][it,:-1], color=cmap(t[it]/tmax))
    


fig, axes = plt.subplots(2,1, figsize=(6,8))
plt.subplots_adjust(hspace=0,wspace=0.1)

basedir = "/global/project/projectdirs/m3761/PAPER2/"
dirlist = ["Fiducial_3D", "convergence/Fiducial_3D_32d"]
data = [h5py.File(basedir+thisdir+"/reduced_data_fft_power.h5","r") for thisdir in dirlist]

for i in range(2):
    makeplot(axes[i],data[i])

# colorbar
cax = fig.add_axes([0.125, .9, .775, .03])
cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
norm = mpl.colors.Normalize(vmin=0, vmax=5)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,orientation="horizontal")
cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cax.minorticks_on()
cax.xaxis.set_minor_locator(MultipleLocator(0.25))

for ax in axes:
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_ylim(1e40,9.9e71)
    ax.set_xlim(-0.04,8)
    
axes[0].set_xticklabels([])
axes[0].set_xlabel(r"$k\,(\mathrm{cm}^{-1})$")

    
axes[0].set_ylabel(r"$|\widetilde{N}_{ee}|^2\,(\mathrm{cm}^{-2})$")
axes[1].set_ylabel(r"$|\widetilde{N}_{ee}|^2\,(\mathrm{cm}^{-2})$")
axes[0].text(0.5,0.85,"Fiducial\_3D",transform=axes[0].transAxes)
axes[1].text(0.5,0.85,"Fiducial\_3D\_32d",transform=axes[1].transAxes)

plt.savefig("power_spectrum_convergence.pdf", bbox_inches='tight')
