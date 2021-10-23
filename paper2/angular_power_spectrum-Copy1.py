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

def makeplot(axes, data,ncells):

    t = np.array(data["t"])
    spectra = np.array(data["angular_spectrum"])*ncells
    for icomponent in range(2):
        this_spectra = spectra[:,:,0,icomponent]
        nl = np.shape(this_spectra)[1]
        l = np.array(range(nl))
        for it in range(len(t)):
            axes[icomponent].semilogy(l, this_spectra[it,:], color=cmap(t[it]/tnorm))
    
def makeplot_lowres(axes, data,ncells):

    t = np.array(data["t"])
    spectra = np.array(data["angular_spectrum"])*ncells
    for icomponent in range(2):
        this_spectra = spectra[:,:,0,icomponent]
        nl = np.shape(this_spectra)[1]
        l = np.array(range(nl))
        it = len(t)-1
        for it in range(len(t)):
            axes[icomponent].semilogy(l, this_spectra[it,:], color=cmap(t[it]/tnorm),linestyle="--", linewidth=0.75)
    

fig, axes = plt.subplots(2,3, figsize=(16,8),sharex=True)
plt.subplots_adjust(hspace=0,wspace=0.1)

######
# 1D #
######
ncells = 128
data = h5py.File("/global/project/projectdirs/m3761/PAPER2/90Degree_1D/reduced_data_angular_power_spectrum.h5","r")
makeplot(axes[:,0], data,ncells)

data = h5py.File("/global/project/projectdirs/m3761/PAPER2/convergence/90Degree_1D_32d/reduced_data_angular_power_spectrum.h5","r")
makeplot_lowres(axes[:,0], data,ncells)

######
# 2D #
######
ncells = 128*128
data = h5py.File("/global/project/projectdirs/m3761/PAPER2/90Degree_2D_outplane/reduced_data_angular_power_spectrum.h5","r")
makeplot(axes[:,1], data,ncells)

data = h5py.File("/global/project/projectdirs/m3761/PAPER2/convergence/90Degree_2D_outplane_32d/reduced_data_angular_power_spectrum.h5","r")
makeplot_lowres(axes[:,1], data,ncells)

######
# 3D #
######
ncells = 128*128*128
data = h5py.File("/global/project/projectdirs/m3761/PAPER2/90Degree_3D/reduced_data_angular_power_spectrum.h5","r")
makeplot(axes[:,2], data,ncells)

data = h5py.File("/global/project/projectdirs/m3761/PAPER2/convergence/90Degree_3D_32d/reduced_data_angular_power_spectrum.h5","r")
makeplot_lowres(axes[:,2], data,ncells)


# colorbar
cax = fig.add_axes([0.92, .11, .02, .77])
cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
norm = mpl.colors.Normalize(vmin=0, vmax=5)
cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax)
cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
cax.minorticks_on()
cax.yaxis.set_minor_locator(MultipleLocator(0.25))

# legend
custom_lines = [Line2D([0],[0], color="k",linestyle="-"),
                Line2D([0],[0], color="k",linestyle="--")]
axes[0,0].legend(custom_lines, [r"$N_\mathrm{eq}=64$",r"$N_\mathrm{eq}=32$"], loc="upper right", frameon=False, fontsize=18)

for ax in axes.flat:
    ax.set_xlim(0,32)
    
for ax in axes[0,:]:
    ax.set_ylim(1.01e59,1e67)

for ax in axes[1,:]:
    ax.set_ylim(1.01e51,1e67)

for ax in axes[:,1:].flat:
    ax.set_yticklabels([])

for ax in axes.flat:
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
# axis labels
for ax in axes[1,:]:
    ax.set_xlabel(r"$l$")

axes[0,0].set_ylabel(r"$|f_l|_{ee}^2$")
axes[1,0].set_ylabel(r"$|f_l|_{e\mu}^2$")
x=5
y=1e66
axes[0,0].text(x,y,"90Degree\_1D")
axes[0,1].text(x,y,"90Degree\_2D")
axes[0,2].text(x,y,"90Degree\_3D")

plt.savefig("angular_power_spectrum.pdf", bbox_inches='tight')
