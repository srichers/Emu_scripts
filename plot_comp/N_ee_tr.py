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
def plotdata(filename,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_avg_mag"])[:,a,b]
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

##############
# formatting #
##############
ax.axhline(1./2., color="green")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
#ax.grid(which='both')

#############
# plot data #
#############

#fid:
filename_emu_2f = "/global/project/projectdirs/m3761/FLASH/Emu/Fiducial_3D_2F/reduced_data.h5"
filename_emu_3f = "/global/project/projectdirs/m3761/FLASH/Emu/Fiducial_3D_3F/reduced_data.h5"
filename_bang = "/global/project/projectdirs/m3761/FLASH/FFI_3D/fid/sim1/reduced_data_nov4_test_hdf5_chk.h5"

#90d:
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_3F/reduced_data.h5"
#filename_bang = "/global/project/projectdirs/m3761/FLASH/FFI_3D/90d/sim1/reduced_data_nov4_test_hdf5_chk.h5"

t,N = plotdata(filename_emu_2f,0,0)
t_ex,N_ex = plotdata(filename_emu_2f,0,1)
tmax = t[np.argmax(N_ex)]
ax.plot(t-tmax, N, 'k-', label=r'${\rm emu\,\,(2f)}$')

t,N = plotdata(filename_emu_3f,0,0)
t_ex,N_ex = plotdata(filename_emu_3f,0,1)
tmax = t[np.argmax(N_ex)]
#special code for excising a single point
bad_ind = 152
#print(N[:])
t = np.concatenate((t[:bad_ind-1], t[bad_ind+1:]))
N = np.concatenate((N[:bad_ind-1], N[bad_ind+1:]))
ax.plot(t-tmax, N, 'k--', label=r'${\rm emu\,\,(3f)}$')

t,N = plotdata(filename_bang,0,0)
t_ex,N_ex = plotdata(filename_bang,0,1)
tmax = t[np.argmax(N_ex)]
ax.plot(t-tmax, N, 'r-', label=r'${\rm FLASH\,\,(2f)}$')
ax.set_xlabel(r"$t-t_{\rm max}\,(10^{-9}{\rm s})$")
ax.set_ylabel(r"$\langle N_{ee}/{\rm Tr}[N]\rangle$")
ax.legend(loc='upper right')
plt.savefig("N_ee_tr_comp_fid.pdf", bbox_inches="tight")
#plt.savefig("N_ee_tr_comp_90d.pdf", bbox_inches="tight")
