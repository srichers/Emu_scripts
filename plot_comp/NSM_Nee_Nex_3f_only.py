# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

n_nue0 = 1.421954234999705e+33     # 1/ccm
n_nux0 = 1.9645407875568215e+33/4. # 1/ccm, each flavor
n_tot = n_nue0 + 2.*n_nux0
n_2F = n_nue0 + n_nux0
n_tot_eq = n_tot/3.0
n_2F_eq = n_2F/2.0
mfact = 1.e-33
mfactstr = r'$10^{-33}\times$'

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
#plt.subplots_adjust(vspace=1)

##############
# formatting #
##############
axes[0].axhline(mfact*n_2F_eq, color="green")
axes[0].axhline(mfact*n_tot_eq, color="green", linestyle='--')
axes[1].set_xlabel(r"$t-t_{\rm max}\,(10^{-9}\,\mathrm{s})$")
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
#axes[0].set_xlim(-1.0, 4.0)
axes[0].set_xlim(-0.5, 2.0)
axes[0].set_ylabel(mfactstr + r"$\langle N_{ee}\rangle\,({\rm cm}^{-3})$")
axes[1].set_ylabel(mfactstr + r"$\langle |N_{ex}|\rangle\,({\rm cm}^{-3})$")
axes[0].set_ylim(0.9*mfact*n_nux0, 1.1*mfact*n_nue0)

#############
# plot data #
#############
#filename_emu_2f = "/global/project/projectdirs/m3761/Evan/merger_2F/reduced_data.h5"
#filename_emu_2f_lowres = "/global/project/projectdirs/m3761/Evan/merger_2F_lowres/reduced_data.h5"
filename_emu_2f = "/global/project/projectdirs/m3761/FLASH/Emu/merger_2F/reduced_data_normalized.h5"
filename_emu_2f_lowres = "/global/cfs/projectdirs/m3761/FLASH/Emu/merger_2F_lowres/reduced_data.h5"

#filename_emu_3f = "/global/project/projectdirs/m3761/Evan/merger_3F/reduced_data.h5"
#filename_emu_3f_lowres = "/global/project/projectdirs/m3761/Evan/merger_3F_lowres/reduced_data.h5"
filename_emu_3f = "/global/project/projectdirs/m3761/FLASH/Emu/merger_3F/reduced_data.h5"
filename_emu_3f_lowres = "/global/project/projectdirs/m3761/FLASH/Emu/merger_3F_lowres/reduced_data.h5"

filename_bang = "/global/project/projectdirs/m3761/FLASH/FFI_3D/NSM_1/sim1/reduced_data_NSM_sim_hdf5_chk.h5"
filename_bang_res1 = "/global/project/projectdirs/m3761/FLASH/FFI_3D/NSM_1/res_test1/from_payne/v0/reduced_data_NSM_sim.h5"
filename_bang_res3 = "/global/project/projectdirs/m3761/FLASH/FFI_3D/NSM_1/res_test3/reduced_data.h5"

t,Nee = plotdata(filename_emu_2f_lowres,0,0)
tex,Nex = plotdata(filename_emu_2f_lowres,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'k-', alpha=0.5)
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'k-', alpha=0.5)

t,Nee = plotdata(filename_emu_2f,0,0)
tex,Nex = plotdata(filename_emu_2f,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'k-', label=r'${\rm {\tt EMU}\,\,(2f)}$')
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'k-', label=r'${\rm {\tt EMU}\,\,(2f)}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('EMU (2F)')
print('N_ee:', n_2F*Nee[a_time], 'N_ex:', n_2F*Nex[np.argmax(Nex)])


t,Nee = plotdata(filename_emu_3f_lowres,0,0)
tex,Nex = plotdata(filename_emu_3f_lowres,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_tot, 'k--', alpha=0.5)
axes[1].semilogy(t-tmax, mfact * Nex * n_tot, 'k--', alpha=0.5)

t,N = plotdata(filename_emu_3f,0,0)
tex,Nex = plotdata(filename_emu_3f,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * N * n_tot, 'k--', label=r'${\rm {\tt EMU}\,\,(3f)}$')
axes[1].semilogy(t-tmax, mfact * Nex * n_tot, 'k--', label=r'${\rm {\tt EMU}\,\,(3f)}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('EMU (3F)')
print('N_ee:', n_tot*Nee[a_time], 'N_ex:', n_tot*Nex[np.argmax(Nex)])


t,Nee = plotdata(filename_bang_res3,0,0)
tex,Nex = plotdata(filename_bang_res3,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'r-', alpha=0.25)
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'r-', alpha=0.25)

t,Nee = plotdata(filename_bang_res1,0,0)
tex,Nex = plotdata(filename_bang_res1,0,1)
tmax = t[np.argmax(Nex)]
#No need for n_2F/n_tot scaling for this data set:
axes[0].plot(t-tmax, mfact * Nee * n_tot, 'r-', alpha=0.5)
axes[1].semilogy(t-tmax, mfact * Nex * n_tot, 'r-', alpha=0.5)

t,Nee = plotdata(filename_bang,0,0)
tex,Nex = plotdata(filename_bang,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'r-', label=r'${\rm{\tt FLASH}\,\,(2f)}$')
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'r-', label=r'${\rm{\tt FLASH}\,\,(2f)}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('FLASH')
print('N_ee:', n_2F*Nee[a_time], 'N_ex:', n_2F*Nex[np.argmax(Nex)])


axes[0].legend(loc=(0.43,0.6), frameon=False)
plt.savefig("NSM_Nee_Nex_3f_only.pdf", bbox_inches="tight")
