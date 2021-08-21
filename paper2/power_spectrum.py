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


def makeplot(axes,data):

    # get appropriate data
    t=data["t"]
    print([t[i+1]-t[i] for i in range(len(t)-1)])
    k=data["k"]
    for it in range(1,len(t)):
        axes[0].semilogy(k, data["N00_FFT"][it,:-1], color=cmap(t[it]/tmax))
        axes[1].semilogy(k, data["N01_FFT"][it,:-1], color=cmap(t[it]/tmax))
    


fig, axes = plt.subplots(2,3, figsize=(16,8))
plt.subplots_adjust(hspace=0,wspace=0.1)

basedir = "/global/project/projectdirs/m3761/PAPER2/"
dirlist = ["Fiducial_3D", "90Degree_3D", "TwoThirds_3D/1"]
data = [h5py.File(basedir+thisdir+"/reduced_data_fft_power.h5","r") for thisdir in dirlist]

for i in range(3):
    makeplot(axes[:,i],data[i])

# colorbar
cax = fig.add_axes([0.92, .11, .02, .77])
cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
norm = mpl.colors.Normalize(vmin=0, vmax=5)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax)
cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
cax.minorticks_on()
cax.yaxis.set_minor_locator(MultipleLocator(0.25))

for ax in axes.flat:
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


for ax in axes[:,1:].flatten():
    ax.set_yticklabels([])

for ax in axes[0,:]:
    ax.set_xticklabels([])
    ax.set_ylim(1e40,1e75)

for ax in axes[1,:]:
    ax.set_ylim(1e50,1e73)
    
for ax in axes[1,:]:
    ax.set_xlabel(r"$k\,(\mathrm{cm}^{-1})$")

for ax in axes[:,:2].flatten():
    ax.set_xlim(-0.04,8)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    
for ax in axes[:,2].flatten():
    ax.set_xlim(-0.01,2)
    
axes[0,0].set_ylabel(r"$|\widetilde{N}_{ee}|^2\,(\mathrm{cm}^{-2})$")
axes[1,0].set_ylabel(r"$|\widetilde{N}_{e\mu}|^2\,(\mathrm{cm}^{-2})$")
axes[0,0].text(0.5,0.85,"Fiducial",transform=axes[0,0].transAxes)
axes[0,1].text(0.5,0.85,"90Degree",transform=axes[0,1].transAxes)
axes[0,2].text(0.5,0.85,"TwoThirds",transform=axes[0,2].transAxes)

plt.savefig("power_spectrum.pdf", bbox_inches='tight')
