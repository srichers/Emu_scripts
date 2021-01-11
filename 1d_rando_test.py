# generate time-series plot
# Run from PAPER/1D/1D_fiducial_long

import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = truncate_colormap(mpl.cm.gist_heat,0,.9)

######################
# read averaged data #
######################
def plotdata(filename):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_avg_mag"])[:,0,0]
    Nbar=np.array(avgData["Nbar_avg_mag"])[:,0,0]
    avgData.close()
    return t, N, Nbar

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


fig, ax = plt.subplots(1,1, figsize=(6,5))
plt.subplots_adjust(hspace=0,wspace=0)

##############
# plot ndens #
##############
basedir = "/global/project/projectdirs/m3018/Emu/PAPER/1D/rando_test"
dirlist = [
    basedir+"/0.0_thirds",
    basedir+"/0.1_thirds",
    basedir+"/0.2_thirds",
    basedir+"/0.3_thirds",
    basedir+"/0.4_thirds",
    basedir+"/0.5_thirds",
    basedir+"/0.6_thirds",
    basedir+"/0.7_thirds",
    basedir+"/0.8_thirds",
    basedir+"/0.9_thirds",
    basedir+"/1.0_thirds",
]
value_list = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]

for value,dirname in zip(value_list,dirlist):
    t,N,Nbar = plotdata(dirname+"/reduced_data.h5")
    print(value)
    ax.plot(t, N,color=cmap(value))
    ax.plot(t, Nbar,color=cmap(value), linestyle="--")

##############
# formatting #
##############
ax.axhline(1./3., color="green")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.set_ylabel(r"$\langle N_{ee}\rangle /\mathrm{Tr}(N)$")
#ax.set_xlim(.1,5)
#ax.set_xscale("log")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# colorbar
ax = fig.add_axes([0,0,0,0])
a = np.array([[0,value_list[-1]]])
img = plt.imshow(a,cmap=cmap, vmax=value_list[-1])
plt.gca().set_visible(False)
cax = fig.add_axes([.95, .11, .06, .78])
cax.tick_params(axis='both', which='both', direction='in')
cbar = plt.colorbar(cax=cax,ticks=value_list)
cbar.set_label(r"$F_{\bar{\nu}}/F_{\nu}$")




############
# save pdf #
############
plt.savefig("1d_rando_test.pdf", bbox_inches="tight")
