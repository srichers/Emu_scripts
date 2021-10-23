# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

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
def plotdataN(filename):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_avg_mag"])[:,0,1]
    avgData.close()
    return t, N

def plotdataF(filename,q):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["F_avg_mag"])[:,q,0,1]
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


fig, axes = plt.subplots(3,4, figsize=(15,15))
plt.subplots_adjust(hspace=0,wspace=0)

##############
# formatting #
##############
for ax in axes.flat:
    ax.set_ylabel(r"$\langle \rho_{e\mu}\rangle$")
    ax.set_xlim(0,5)
    #ax.set_xscale("log")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
def slope(t,N):
    loc1 = np.where(N>1e-4)[0][0]
    loc2 = np.where(N>1e-2)[0][0]
    print( np.log(N[loc2]/N[loc1]) / (t[loc2]-t[loc1]) )

#############
# plot data #
#############
basedir = "/global/project/projectdirs/m3761/PAPER2"
dirlist    = [["-", "gray" , "1D", basedir+"/Fiducial_1D"],
              ["-", "black" , "2D", basedir+"/Fiducial_2D"],
              ["-", "blue" , "3D", basedir+"/Fiducial_3D"],]
print("Fiducial")
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data.h5"
    t,N = plotdataN(filename)
    axes[0,0].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    slope(t,N)
    t,N = plotdataF(filename,0)
    axes[0,1].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    t,N = plotdataF(filename,1)
    axes[0,2].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    t,N = plotdataF(filename,2)
    axes[0,3].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])


dirlist    = [["-" , "gray" , "1D"               , basedir+"/90Degree_1D"],
              #["-" , "black", "2D (in plane)"    , "ocean/projects/phy200048p/shared/2D/90deg_inplane"],
              ["-", "black", "2D", basedir+"/90Degree_2D_outplane"],
              ["-" , "blue" , "3D"               , basedir+"/90Degree_3D"]]
print("90Degree")
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data.h5"
    t,N = plotdataN(filename)
    axes[1,0].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    slope(t,N)
    t,N = plotdataF(filename,0)
    axes[1,1].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    t,N = plotdataF(filename,1)
    axes[1,2].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    t,N = plotdataF(filename,2)
    axes[1,3].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])

dirlist    = [["-", "gray", "1D", basedir+"/TwoThirds_1D"],
              ["-", "black", "2D", basedir+"/TwoThirds_2D"],
              ["-", "blue", "3D", basedir+"/TwoThirds_3D/1"]]
print("TwoThirds")
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data.h5"
    t,N = plotdataN(filename)
    axes[2,0].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    slope(t,N)
    t,N = plotdataF(filename,0)
    axes[2,0].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    t,N = plotdataF(filename,1)
    axes[2,2].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])
    t,N = plotdataF(filename,2)
    axes[2,3].semilogy(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])


axes[0,0].legend(frameon=False,ncol=1,fontsize=18, loc=(.4,.3))
#axes[1].legend(frameon=False,ncol=1,fontsize=18, loc=(.4,.3))
#axes[2].legend(frameon=False,ncol=1,fontsize=18, loc=(.4,.3))
axes[0,0].set_xticklabels([])
axes[1,0].set_xticklabels([])
#axes[0].text(2.5,.9,"Fiducial")
#axes[1].text(2.5,.9,"90Degree")
#axes[2].text(2.5,.9,"TwoThirds")
#axes[2].set_ylim(.6,1.05)
axes[2,0].set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")


############
# save pdf #
############
plt.savefig("avgfemu.pdf", bbox_inches="tight")
