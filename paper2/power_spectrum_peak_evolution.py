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


def makeplot(ax,data, ls, label, color):

    # get appropriate data
    t=np.array(data["t"])
    k=data["k"]
    kpeak = []
    for it in range(len(t)):
        fft = data["N01_FFT"][it,:-1]
        kmax = k[np.argmax(fft)]
        kpeak.append(kmax)
    kpeak = np.array(kpeak)
    print(np.shape(kpeak))
    ax.plot(t*1e9, kpeak, linestyle=ls, color=color, label=label)
    


fig, ax = plt.subplots(1,1, figsize=(6,5))
plt.subplots_adjust(hspace=0,wspace=0.1)

basedir = "/global/project/projectdirs/m3761/PAPER2/"
dirlist = ["Fiducial_1D","convergence/Fiducial_2D_32d","Fiducial_2D","convergence/Fiducial_2D_128d","convergence/Fiducial_2D_256d","Fiducial_3D", "convergence/Fiducial_3D_32d"]
#dirlist = ["90Degree_1D","90Degree_2D_outplane","convergence/90Degree_2D_outplane_32d","90Degree_3D", "convergence/90Degree_3D_32d"]
data = [h5py.File(basedir+thisdir+"/reduced_data_fft_power.h5","r") for thisdir in dirlist]
linestyles = ["-","--","-","--","--","-","--"]
colors = ["gray","black","black","green","red","blue","blue"]
labels = ["Fiducial\_1D","Fiducial\_2D\_32d","Fiducial\_2D","Fiducial\_2D\_128d","Fiducial\_2D\_256d","Fiducial\_3D","Fiducial\_3D\_32d"]

for i in range(len(dirlist)):
    makeplot(ax,data[i], linestyles[i], labels[i], colors[i])

ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.set_ylabel(r"$k_\mathrm{peak}\,(\mathrm{cm}^{-1})$")
ax.set_xlim(0,5)
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.legend(frameon=False, loc="upper right", fontsize=16)
ax.minorticks_on()

plt.savefig("power_spectrum_peak_evolution.pdf", bbox_inches='tight')
