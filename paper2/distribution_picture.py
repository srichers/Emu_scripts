import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

################
# plot options #
################
mpl.rcParams['font.size'] = 40
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


fig, axes = plt.subplots(3,1, figsize=(6,22), subplot_kw={'projection': 'polar'})
plt.subplots_adjust(hspace=0,wspace=0)

theta = np.arange(0,2.01*np.pi,np.pi/100.)
lw = 3

axes[0].plot(theta, 1.+np.cos(theta - np.pi/2.), color='blue', lw=lw,label=r"$\nu_e$")
axes[0].plot(theta, 1.+np.cos(theta + np.pi/2.), color='red', lw=lw, label=r"$\bar{\nu}_e$")
axes[0].text(-np.pi, 2.5, "Fiducial",ha='center',va='center',rotation=90)
axes[0].legend(frameon=False, loc=(0.7,-0.3))

axes[1].plot(theta, 1.+np.cos(theta - np.pi/2.), color='blue', lw=lw)
axes[1].plot(theta, 1.+np.cos(theta           ), color='red',lw=lw)
axes[1].text(-np.pi, 2.5, "90Degree",ha='center',va='center',rotation=90)

axes[2].plot(theta, np.ones(theta.shape)       , color='blue',lw=lw)
axes[2].plot(theta, 2./3.*(1.+np.cos(theta + np.pi/2.)), color='red',lw=lw)
axes[2].text(-np.pi, 2.5, "TwoThirds",ha='center',va='center',rotation=90)


for ax in axes:
    ax.grid(linewidth=1)
    ax.set_rticks([1,2])
    ax.set_yticklabels([])
    ax.set_xticks([0, np.pi/2., np.pi, 3.*np.pi/2.])
    ax.set_xticklabels([r"$\hat{x}$",r"$\hat{z}$","",""])
    ax.spines['polar'].set_visible(False)
    ax.set_rlim(0,2.1)

plt.savefig("distribution_picture.pdf", bbox_inches="tight")
