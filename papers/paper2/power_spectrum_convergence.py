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
    k=np.array(data["k"])*2.*np.pi
    for it in range(1,len(t)):
        ax.semilogy(k, data["N01_FFT"][it,:-1], color=cmap(t[it]/tmax))
    
def makeplot_last(ax, data):
    # get appropriate data
    t=np.array(data["t"])
    k=np.array(data["k"])*2.*np.pi
    it = np.argmax(t>1e-9) # index of t=2ns
    ax.semilogy(k, data["N01_FFT"][it,:-1], color="k",linestyle="--")


fig, axes = plt.subplots(2,1, figsize=(6,8))
plt.subplots_adjust(hspace=0,wspace=0.1)

basedir = "/global/project/projectdirs/m3761/PAPER2/"
dirlist = ["convergence/Fiducial_3D_32d", "Fiducial_3D"]
data = [h5py.File(basedir+thisdir+"/reduced_data_fft_power.h5","r") for thisdir in dirlist]

for i in range(2):
    makeplot(axes[i],data[i])
#for ax in axes:
#    for i in range(len(dirlist)):
#        makeplot_last(ax,data[i])

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
    ax.set_ylim(9.9e50,9.9e70)
    ax.set_xlim(-0.04,8*2.*np.pi)
    
axes[0].set_xticklabels([])
axes[1].set_xlabel(r"$|k|\,(\mathrm{cm}^{-1})$")

    
axes[0].set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-6})$")
axes[1].set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-6})$")
axes[1].text(0.5,0.85,"Fiducial\_3D",transform=axes[1].transAxes)
axes[0].text(0.5,0.85,"Fiducial\_3D\_32d",transform=axes[0].transAxes)

plt.savefig("power_spectrum_convergence.pdf", bbox_inches='tight')
