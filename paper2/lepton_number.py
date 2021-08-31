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
def plotdata(filename,weight,weightbar):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_avg_mag"])[:,0,0]*weight
    Nbar=np.array(avgData["Nbar_avg_mag"])[:,0,0]*weightbar
    avgData.close()
    return t, ((N-Nbar)-(weight-weightbar))/(weight+weightbar)*1e6

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
# formatting #
##############
ax.set_ylabel(r"$10^6\times$ ELN Relative Error")
ax.set_xlim(0,5)
#ax.set_xscale("log")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

#############
# plot data #
#############
basedir = "/global/project/projectdirs/m3761/PAPER2"
dirlist    = [["-", "gray" , "1D", basedir+"/Fiducial_1D"],
              ["-", "black" , "2D", basedir+"/Fiducial_2D"],
              ["-", "blue" , "3D", basedir+"/Fiducial_3D"],]
#axes[0].axhline(0, color="green")
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data.h5"
    t,N = plotdata(filename,1,1)
    #badlocs = np.where(N<.2)
    #print(badlocs)
    #N[badlocs]=0.345
    ax.plot(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])


dirlist    = [["--" , "gray" , ""               , basedir+"/90Degree_1D"],
              #["-" , "black", "2D (in plane)"    , "ocean/projects/phy200048p/shared/2D/90deg_inplane"],
              ["--", "black", "90Degree", basedir+"/90Degree_2D_outplane"],
              ["--" , "blue" , ""               , basedir+"/90Degree_3D"]]
#axes[1].axhline(1./3., color="green")
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data.h5"
    t,N = plotdata(filename,1,1)
    ax.plot(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])

dirlist    = [[":", "gray", "", basedir+"/TwoThirds_1D"],
              [":", "black", "TwoThirds", basedir+"/TwoThirds_2D"],
              [":", "blue", "", basedir+"/TwoThirds_3D/1"]]
#axes[2].axhline(0.7, color="green")
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data.h5"
    t,N = plotdata(filename,1,2./3.)
    ax.plot(t, N,color=inputs[1],linestyle=inputs[0],label=inputs[2])


ax.legend(frameon=False,ncol=1,fontsize=18, loc="lower left")
#axes[1].legend(frameon=False,ncol=1,fontsize=18, loc=(.4,.3))
#axes[2].legend(frameon=False,ncol=1,fontsize=18, loc=(.4,.3))
#axes[2].set_ylim(.6,1.05)
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")


############
# save pdf #
############
plt.savefig("lepton_number.pdf", bbox_inches="tight")
