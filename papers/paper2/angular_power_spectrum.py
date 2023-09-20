import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from matplotlib.lines import Line2D

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

component_list = ["00","01","02","11","12","22"]

tnorm = 5e-9

def makeplot(ax, data,ncells):

    t = np.array(data["t"])
    spectra = np.array(data["angular_spectrum"])*ncells
    this_spectra = spectra[:,:,0,1]
    nl = np.shape(this_spectra)[1]
    l = np.array(range(nl))
    for it in range(len(t)):
        ax.semilogy(l, this_spectra[it,:], color=cmap(t[it]/tnorm))
    ax.semilogy(l, this_spectra[-1,:],color="k")
    
def makeplot_lowres(ax, data,ncells):

    t = np.array(data["t"])
    spectra = np.array(data["angular_spectrum"])*ncells
    this_spectra = spectra[:,:,0,1]
    nl = np.shape(this_spectra)[1]
    l = np.array(range(nl))
    it = len(t)-1
    for it in range(len(t)):
        ax.semilogy(l, this_spectra[it,:], color=cmap(t[it]/tnorm),linestyle="--", linewidth=0.75)
    ax.semilogy(l, this_spectra[-1,:], color="k",linestyle="--")
    

fig, axes = plt.subplots(2,1, figsize=(6,12),sharex=True)
plt.subplots_adjust(hspace=0,wspace=0.1)

######
# 1D #
######
ncells = 128
data = h5py.File("/global/project/projectdirs/m3761/PAPER2/90Degree_1D/reduced_data_angular_power_spectrum.h5","r")
makeplot(axes[0], data,ncells)

data = h5py.File("/global/project/projectdirs/m3761/PAPER2/convergence/90Degree_1D_32d/reduced_data_angular_power_spectrum.h5","r")
makeplot_lowres(axes[0], data,ncells)

######
# 2D #
######
#ncells = 128*128
#data = h5py.File("/global/project/projectdirs/m3761/PAPER2/90Degree_2D_outplane/reduced_data_angular_power_spectrum.h5","r")
#makeplot(axes[:,1], data,ncells)

#data = #h5py.File("/global/project/projectdirs/m3761/PAPER2/convergence/90Degree_2D_outplane_32d/reduced_data_angular_power_spectrum.h5","r")
#makeplot_lowres(axes[:,1], data,ncells)

######
# 3D #
######
ncells = 128*128*128
data = h5py.File("/global/project/projectdirs/m3761/PAPER2/90Degree_3D/reduced_data_angular_power_spectrum.h5","r")
makeplot(axes[1], data,ncells)

data = h5py.File("/global/project/projectdirs/m3761/PAPER2/convergence/90Degree_3D_32d/reduced_data_angular_power_spectrum.h5","r")
makeplot_lowres(axes[1], data,ncells)


# colorbar
cax = fig.add_axes([0.125, .9, .77, .02])
cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
norm = mpl.colors.Normalize(vmin=0, vmax=5)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,orientation="horizontal")
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')
cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
cax.minorticks_on()
cax.xaxis.set_minor_locator(MultipleLocator(0.25))

# legend
custom_lines = [Line2D([0],[0], color="k",linestyle="-"),
                Line2D([0],[0], color="k",linestyle="--")]
axes[0].legend(custom_lines, [r"$N_\mathrm{eq}=64$",r"$N_\mathrm{eq}=32$"], loc="upper right", frameon=False, fontsize=18)

for ax in axes:
    ax.set_xlim(0,32)
    ax.set_ylim(1.01e51,1e68)
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(LogLocator(base=100,subs=(1,),numticks=20))
    ax.set_ylabel(r"$|f_{e\mu,l}|^2$")
    
# axis labels
axes[1].set_xlabel(r"$l$")
x=5
y=1e66
axes[0].text(x,y,"90Degree\_1D")
axes[1].text(x,y,"90Degree\_3D")

plt.savefig("angular_power_spectrum.pdf", bbox_inches='tight')
