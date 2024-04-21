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

def plotdata_new_format(filename,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t(s)"])*1e9
    N=np.array(avgData["N_avg_mag(1|ccm)"])[:,a,b]
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
#plt.subplots_adjust(vspace=1)

#############
# plot data #
#############
filename_1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t2/sim2/reduced_data.h5"
filename_2 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/sim/reduced_data.h5"
labels = [r"$H_M=0$", r"$H_M\ne0$"]
namestr = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/comp_t/Nee_Nex_tr_2comp.pdf"
xlims = (-0.5, 0.5)
#2/3:
#box_length = 32.0
#n_grid = 128

t_1,Nee_1 = plotdata(filename_1,0,0)
tex_1,Nex_1 = plotdata(filename_1,0,1)
txx_1,Nxx_1 = plotdata(filename_1,1,1)
n_2F = Nee_1[0] + Nxx_1[0]
tmax_1 = t_1[np.argmax(Nex_1)]
axes[0].plot(t_1-tmax_1, Nee_1 / n_2F, 'r-', label=labels[0])
n_2F_eq = n_2F/2.0
axes[0].axhline(n_2F_eq, color="green")
axes[1].semilogy(t_1-tmax_1, Nex_1 / n_2F, 'r-')

#t_1,Nee_1 = plotdata_new_format(filename_1,0,0)
#tex_1,Nex_1 = plotdata_new_format(filename_1,0,1)
#txx_1,Nxx_1 = plotdata_new_format(filename_1,1,1)
#n_2F = Nee_1[0] + Nxx_1[0]
#n_2F_eq = n_2F/2.0
#tmax_1 = t_1[np.argmax(Nex_1)]
#axes[0].plot(t_1-tmax_1, Nee_1/n_2F, 'k-', label=r'$\texttt{Emu}$')
#axes[0].axhline(0.5, color="green")
#axes[1].semilogy(t_1-tmax_1, Nex_1/n_2F, 'k-')

t_2,Nee_2 = plotdata(filename_2,0,0)
tex_2,Nex_2 = plotdata(filename_2,0,1)
txx_2,Nxx_2 = plotdata(filename_2,1,1)
n_2F = Nee_2[0] + Nxx_2[0]
tmax_2 = t_2[np.argmax(Nex_2)]
axes[0].plot(t_2-tmax_2, Nee_2 / n_2F, 'b--', label=labels[1])
axes[1].plot(t_2-tmax_2, Nex_2 / n_2F, 'b--', label=labels[1])
#axes[1].axhline(5.e-7, color='b', linestyle=':')

#fig.text(0.5, 0.82, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
#fig.text(0.5, 0.77, r'$N_{{gp}}={}^3$'.format(n_grid))
#fig.text(0.55, 0.7, r'$\delta m^2 = 7.53\times10^{-5}\,{\rm eV}^2$')
#fig.text(0.55, 0.65, r'$\theta_{12}=0.587$')


##############
# formatting #
##############
axes[1].set_xlabel(r'$t-t_{\rm sat}\,({\rm s})$')
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
#axes[0].set_xlim(-1.0, 4.0)
axes[0].set_xlim(xlims)
axes[0].set_ylabel(r'$\langle N_{ee}\rangle/{\rm Tr}[N]$')
axes[1].set_ylabel(r'$\langle|N_{ex}|\rangle/{\rm Tr}[N]$')

#axes[0].legend(loc=(0.43,0.1), frameon=False)
axes[0].legend(loc='best', frameon=False)
#plt.savefig("Nee_Nex_tr_2comp.pdf", bbox_inches="tight")
plt.savefig(namestr, bbox_inches="tight")
#plt.savefig("./comp_emu/Nee_Nex_tr_2comp.pdf", bbox_inches="tight")
