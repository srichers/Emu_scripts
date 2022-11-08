# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

n_nue0 = 1.421954234999705e+33     # 1/ccm
n_nux0 = 1.9645407875568215e+33/4. # 1/ccm, each flavor flavors
n_tot = n_nue0 + 2.*n_nux0
n_2F = n_nue0 + n_nux0

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
#mpl.rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
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


fig, axes = plt.subplots(2,1, figsize=(6,10), sharex=True)
plt.subplots_adjust(hspace=0)

##############
# formatting #
##############
axes[0].axhline(1./2., color="green")
axes[-1].set_xlabel(r"$t-t_{\rm max}\,(10^{-9}\,\mathrm{s})$")
for ax in axes:
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    #ax.grid(which='both')
axes[0].set_ylabel(r"$\langle N_{ee}/{\rm Tr}[N]\rangle$")
axes[1].set_ylabel(r"$\langle N_{ex}/{\rm Tr}[N]\rangle$")
axes[0].set_xlim(-1.0, 4.0)

#############
# plot data #
#############
filename_emu_2f = "/global/project/projectdirs/m3761/Evan/merger_2F/reduced_data.h5"
filename_emu_2f_lowres = "/global/project/projectdirs/m3761/Evan/merger_2F_lowres/reduced_data.h5"
filename_emu_3f = "/global/project/projectdirs/m3761/Evan/merger_3F/reduced_data.h5"
filename_emu_3f_lowres = "/global/project/projectdirs/m3761/Evan/merger_3F_lowres/reduced_data.h5"
filename_bang = "/global/project/projectdirs/m3761/FLASH/FFI_3D/NSM/sim1/from_payne/v0/reduced_data_NSM_sim_hdf5_chk.h5"
filename_bang_res1 = "/global/project/projectdirs/m3761/FLASH/FFI_3D/NSM/res_test1/from_payne/v0/reduced_data_NSM_sim.h5"

t,Nee = plotdata(filename_emu_2f_lowres,0,0)
tex,Nex = plotdata(filename_emu_2f_lowres,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee * n_2F/n_tot, 'k-', alpha=0.5)
axes[1].semilogy(t-tmax, Nex * n_2F/n_tot, 'k-', alpha=0.5)

t,Nee = plotdata(filename_emu_2f,0,0)
tex,Nex = plotdata(filename_emu_2f,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee * n_2F/n_tot, 'k-', label=r'${\rm Emu\,\,(2f)}$')
axes[1].semilogy(t-tmax, Nex * n_2F/n_tot, 'k-', label=r'${\rm Emu\,\,(2f)}$')


t,Nee = plotdata(filename_emu_3f_lowres,0,0)
tex,Nex = plotdata(filename_emu_3f_lowres,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee, 'k--', alpha=0.5)
axes[1].semilogy(t-tmax, Nex, 'k--', alpha=0.5)

t,N = plotdata(filename_emu_3f,0,0)
tex,Nex = plotdata(filename_emu_3f,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, N, 'k--', label=r'${\rm Emu\,\,(3f)}$')
axes[1].semilogy(t-tmax, Nex, 'k--', label=r'${\rm Emu\,\,(3f)}$')


t,Nee = plotdata(filename_bang_res1,0,0)
tex,Nex = plotdata(filename_bang_res1,0,1)
tmax = t[np.argmax(Nex)]
#No need for n_2F/n_tot scaling for this data set:
axes[0].plot(t-tmax, Nee, 'r-', alpha=0.5)
axes[1].semilogy(t-tmax, Nex, 'r-', alpha=0.5)

t,Nee = plotdata(filename_bang,0,0)
tex,Nex = plotdata(filename_bang,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee * n_2F/n_tot, 'r-', label=r'${\rm FLASH\,\,(2f)}$')
axes[1].semilogy(t-tmax, Nex * n_2F/n_tot, 'r-', label=r'${\rm FLASH\,\,(2f)}$')


axes[0].legend(loc=(0.43,0.6), frameon=False)
plt.savefig("NSM_Nee_Nex.pdf", bbox_inches="tight")
