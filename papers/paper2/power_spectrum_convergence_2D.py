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

fig, axes = plt.subplots(4,1, figsize=(6,16))
plt.subplots_adjust(hspace=0,wspace=0.1)

basedir = "/global/project/projectdirs/m3761/PAPER2/"
dirlist = ["convergence/Fiducial_2D_32d","Fiducial_2D","convergence/Fiducial_2D_128d","convergence/Fiducial_2D_256d"]

for i in range(len(dirlist)):
    data = h5py.File(basedir+dirlist[i]+"/reduced_data_fft_power.h5","r")
    makeplot(axes[i],data)

# colorbar
cax = fig.add_axes([0.125, .9, .775, .02])
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
    ax.set_ylim(1e50,9.9e71)
    ax.set_xlim(-0.04,8*2.*np.pi)

for ax in axes[:-1]:
    ax.set_xticklabels([])
axes[3].set_xlabel(r"$|k|\,(\mathrm{cm}^{-1})$")

    
axes[0].set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-2})$")
axes[1].set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-2})$")
axes[2].set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-2})$")
axes[3].set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-2})$")
axes[0].text(0.5,0.85,"Fiducial\_2D\_32d",transform=axes[0].transAxes)
axes[1].text(0.5,0.85,"Fiducial\_2D",transform=axes[1].transAxes)
axes[2].text(0.5,0.85,"Fiducial\_2D\_128d",transform=axes[2].transAxes)
axes[3].text(0.5,0.85,"Fiducial\_2D\_256d",transform=axes[3].transAxes)

plt.savefig("power_spectrum_convergence_2D.pdf", bbox_inches='tight')


#######################
#######################
#plt.cla()


def makeplot_last(ax, data, color, linestyle, label):
    # get appropriate data
    t=np.array(data["t"])
    k=np.array(data["k"])*2.*np.pi
    it = np.argmax(t>1e-9) # index of t=2ns
    ax.semilogy(k, data["N01_FFT"][it,:-1], color=color,linestyle=linestyle, label=label)
    

fig, ax = plt.subplots(1,1, figsize=(6,5))

dirlist   = [("Fiducial_1D"        ,"gray", "-" , "Fiducial\_1D"),
    ("convergence/Fiducial_2D_32d" ,"k"   , "--", "Fiducial\_2D\_32d" ),
    ("Fiducial_2D"                 ,"k"   , "-" , "Fiducial\_2D"     ),
    ("convergence/Fiducial_2D_128d","k"   , ":" , "Fiducial\_2D\_128d"),
    ("convergence/Fiducial_2D_256d","k"   , "-.", "Fiducial\_2D\_256d"),
    ("convergence/Fiducial_3D_32d" ,"blue", "--", "Fiducial\_3D\_32d" ),
    ("Fiducial_3D"                 ,"blue", "-" , "Fiducial\_3D"     )]

for dirname, color, ls, label in dirlist:
    makeplot_last(ax, h5py.File(basedir+dirname+"/reduced_data_fft_power.h5","r"), color, ls, label)
    
ax.legend(frameon=False, loc="upper right", fontsize=14)

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
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylim(1e50,9.9e71)
ax.set_xlim(-0.04,8*2.*np.pi)
ax.set_xlabel(r"$|k|\,(\mathrm{cm}^{-1})$")
ax.set_ylabel(r"$|\widetilde{n}_{e\mu}|^2\,(\mathrm{cm}^{-6})$")
    
plt.savefig("power_spectrum_convergence_2D_1ns.pdf", bbox_inches='tight')