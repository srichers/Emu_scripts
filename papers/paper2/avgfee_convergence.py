# Run from /ocean/projects/phy200048p/shared/2D to generate plot showing time evolution of <fee> at different resolutions

import numpy as np
import matplotlib.pyplot as plt
import glob
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
    N=np.array(avgData["N_avg_mag"])[:,0,0]
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


fig, axes = plt.subplots(2,3, figsize=(16,8), sharey=True, sharex=True)
plt.subplots_adjust(hspace=0,wspace=0)
basedir = "/global/project/projectdirs/m3761/PAPER2"

def plot(plotinfo, i, j):
    for dirname,linestyle,label,color in plotinfo:
        filename = dirname+"/reduced_data.h5"
        t,N = plotdata(filename)
        badlocs = np.where(N<0.3)
        N = np.delete(N,badlocs)
        t = np.delete(t,badlocs)
        axes[i,j].plot(t, N,color=color,linestyle=linestyle,label=label)

###############
# Fiducial 3D #
###############
plotinfo = [
    [basedir+"/Fiducial_3D", "-", "Base","black"],
    [basedir+"/convergence/Fiducial_3D_32d", "--", "32d","blue"],
]
plot(plotinfo, 1, 0)

###############
# Fiducial 2D #
###############
plotinfo = [
    [basedir+"/Fiducial_2D", "-", "Base","k"],
    [basedir+"/convergence/Fiducial_2D_32d", "--", "32d","blue"],
    [basedir+"/convergence/Fiducial_2D_128d", "--", "128d","orange"],
    [basedir+"/convergence/Fiducial_2D_256d", "--", "256d","green"],
    [basedir+"/convergence/Fiducial_2D_nx256_32d", "--", "nx256 32d","brown"],
    [basedir+"/convergence/Fiducial_2D_nx256_16cm_32d", "--", "nx256 16cm 32d","magenta"],
    [basedir+"/convergence/Fiducial_2D_nx1024_64cm_16d", "--", "nx1024 64cm 16d","red"],
]
plot(plotinfo, 0, 0)

###############
# 90Degree 3D #
###############
plotinfo = [
    [basedir+"/90Degree_3D", "-", "Base","black"],
    [basedir+"/convergence/90Degree_3D_32d", "--", "32d","blue"],
]
plot(plotinfo, 1, 1)

###############
# 90Degree 2D #
###############
plotinfo = [
    [basedir+"/90Degree_2D_outplane", "-", "Base","black"],
    [basedir+"/convergence/90Degree_2D_outplane_32d", "--", "32d","blue"],
    [basedir+"/convergence/90Degree_2D_outplane_nx1024_64cm_16d", "--", "nx1024 64cm 16d","red"],
    [basedir+"/convergence/90Degree_2D_inplane_32d", "--", "inplane 32d","green"],
    [basedir+"/convergence/90Degree_2D_inplane_nx1024_64cm_16d", "--", "inplane nx1024 64cm 16d","magenta"],
]
plot(plotinfo, 0, 1)

################
# TwoThirds 3D #
################
plotinfo = [
    [basedir+"/TwoThirds_3D/1", "-", "Base","black"],
    [basedir+"/convergence/TwoThirds_3D_32d", "--", "32d","blue"],
]
plot(plotinfo, 1, 2)

################
# TwoThirds 2D #
################
plotinfo = [
    [basedir+"/TwoThirds_2D", "-", "Base","black"],
    [basedir+"/convergence/TwoThirds_2D_32d", "--", "32d","blue"],
    [basedir+"/convergence/TwoThirds_2D_nx256", "--", "nx256","red"],
    [basedir+"/convergence/TwoThirds_2D_nx256_64cm", "--", "nx256 64cm","green"],
]
plot(plotinfo, 0, 2)

##############
# formatting #
##############
for ax in axes.flatten():
    ax.axhline(1./3., color="green")
    ax.set_xlim(0,5)
    ax.set_ylim(0.25,1.05)
    #ax.set_xscale("log")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.grid(which='both')
    ax.minorticks_on()
    ax.legend(frameon=False,ncol=1,fontsize=12, loc=(.35,.2))

#axes[0,0].set_xticklabels([])
axes[0,0].text(2.5,.9,"Fiducial\_2D")
axes[1,0].text(2.5,.9,"Fiducial\_3D")
axes[0,1].text(2.5,.9,"90Degree\_2D")
axes[1,1].text(2.5,.9,"90Degree\_3D")
axes[0,2].text(2.5,.9,"TwoThirds\_2D")
axes[1,2].text(2.5,.9,"TwoThirds\_3D")
for ax in axes[1,:]:
    ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
for ax in axes[:,0]:
    ax.set_ylabel(r"$\langle \rho_{ee}\rangle$")



############
# save pdf #
############
plt.savefig("avgfee_convergence.pdf", bbox_inches="tight")
