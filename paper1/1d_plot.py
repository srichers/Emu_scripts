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


base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]
# real/imag
R=0
I=1
    

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N"])[:,0,0,R]
    avgData.close()
    return t, N

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
ax.axhline(1./3., color="green")

#############
# plot data #
#############
filename = "/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial_long/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="gray",label=r"1D Fiducial")

filename = "/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial_random_IC/0/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="gray",label=r"1D Fiducial")

filename = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_fiducial/0/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="black",label=r"2D Fiducial", linestyle=":")

filename = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_128cm/0/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="black",label=r"2D 128cm", linestyle="--")

filename = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_nx512/0/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="red",label=r"2D nx512", linestyle="--")

filename = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_32dir/0/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="blue",label=r"2D 32dir", linestyle="--")

filename = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_4dir/0/avg_data.h5"
t,N = plotdata(filename)
ax.plot(t, N,color="green",label=r"2D 4dir", linestyle="--")

##############
# formatting #
##############
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.set_ylabel(r"$\langle N_{ij}\rangle /\mathrm{Tr}(N)$")
ax.legend(frameon=False,ncol=1,fontsize=18)
ax.set_xlim(.1,5)
ax.set_xscale("log")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
#ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

############
# save pdf #
############
plt.savefig("1d_fullplot.pdf", bbox_inches="tight")
