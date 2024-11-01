import os
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
def plotdata(filename_avg):
    if not os.path.exists(filename_avg):
        return [0,],[0,]

    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData["t"])
    Nee=np.array(avgData["N_avg_mag"][:,0,0])
    Nxx=np.array(avgData["N_avg_mag"][:,1,1])
    trace = Nee+Nxx
    Nex=np.array(avgData["N_avg_mag"][:,0,1])
    avgData.close()

    return t, (Nex/trace)

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


#############
# plot data #
#############

filename_bang_avg   = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/xy_large/sim/reduced_data.h5"
basename   = "/pscratch/sd/e/egrohs//FFI_3D/MPC/NSM/NSM_3/d_pert/t3/xy_large/sim/volume_rendering/"

t,Nex = plotdata(filename_bang_avg)
t = 1.e+9*t

for i in range(len(t)):

    fig = plt.figure(i, figsize=(10.24, 10.24))
    ax = fig.gca()
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.set_ylabel(r"$\langle |N_{ex}|\rangle/{\rm Tr}[N]$", labelpad=5)
    ax.set_xlabel(r"$t\,({\rm ns})$")

    ax.semilogy(t, Nex, 'r-')
    ax.axvline(x=t[i], color='g')

    plt.savefig(basename + "Nex_1res_tind_{0:04d}.png".format(i))

    fig.clf()
