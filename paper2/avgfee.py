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
#mpl.rc('text', usetex=True)
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


fig, axes = plt.subplots(2,1, figsize=(6,10))
plt.subplots_adjust(hspace=0,wspace=0)

##############
# formatting #
##############
for ax in axes:
    ax.axhline(1./3., color="green")
    ax.set_ylabel(r"$\langle N_{ee}\rangle /\mathrm{Tr}(N)$")
    ax.set_xlim(.1,5)
    #ax.set_xscale("log")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

#############
# plot data #
#############
dirlist    = ["1D/1D_fiducial","2D/fiducial_2D","3D/fiducial_3D"]
linestyles = ["-"             ,"-"             ,"-"             ]
colors     = ["gray"          ,"black"         ,"blue"          ]
labels     = ["1D"            ,"2D"            ,"3D"            ]
for i in range(len(dirlist)):
    filename = "/ocean/projects/phy200048p/shared/"+dirlist[i]+"/reduced_data.h5"
    t,N = plotdata(filename)
    axes[0].plot(t, N,color=colors[i],linestyle=linestyles[i],label=labels[i])


dirlist    = ["1D/90deg","2D/90deg_inplane","2D/90deg_outofplane"]
linestyles = ["-"       ,"-"               ,"--"                 ]
colors     = ["gray"    ,"black"           ,"black"              ]
labels     = ["1D"      ,"2D (in plane)"   ,"2D (out of plane)"  ]
for i in range(len(dirlist)):
    filename = "/ocean/projects/phy200048p/shared/"+dirlist[i]+"/reduced_data.h5"
    t,N = plotdata(filename)
    axes[1].plot(t, N,color=colors[i],linestyle=linestyles[i],label=labels[i])


axes[0].legend(frameon=False,ncol=1,fontsize=18, loc=(.2,.3))
axes[1].legend(frameon=False,ncol=1,fontsize=18, loc=(.2,.3))
axes[0].set_xticklabels([])
axes[0].text(2.5,.9,"Fiducial")
axes[1].text(2.5,.9,"90 Degree")
axes[1].set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")


############
# save pdf #
############
plt.savefig("avgfee.pdf", bbox_inches="tight")
