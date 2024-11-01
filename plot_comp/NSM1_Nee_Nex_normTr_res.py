
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

base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]
# real/imag
R=0
I=1
    
t_str = ["t", "t(s)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]


def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename,a,b,ind):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData[t_str[ind]])*1e9
    N=np.array(avgData[N_str[ind]])[:,a,b]
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
axes[0].set_title(r'${\rm NSM1}$')

##############
# formatting #
##############
axes[0].axhline(y=0.5, color="green", label=None)
axes[1].set_xlabel(r"$t-t_{\rm sat}\,(10^{-9}\,\mathrm{s})$")
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
#axes[0].set_xlim(-1.0, 4.0)
axes[0].set_xlim(-0.6, 1.0)
axes[0].set_ylabel(r"$\langle N_{ee}\rangle/\langle{\rm Tr}[N]\rangle$")
axes[1].set_ylabel(r"$\langle |N_{ex}|\rangle/\langle{\rm Tr}[N]\rangle$")
#axes[1].set_ylim(1.e-8, 1.0)

#############
# plot data #
#############

filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/merger_2F/reduced_data_normalized.h5"
filename_emu_2f_lowres = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/merger_2F_lowres/reduced_data.h5"

filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/sim/reduced_data.h5"
filename_bang_res1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/res_a/reduced_data.h5"
filename_bang_res2 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/res_b/reduced_data.h5"

savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/comp_emu/"

ind = np.zeros([3], dtype=np.int8)


t,Nxx = plotdata(filename_emu_2f_lowres,1,1, ind[0])
t,Nee = plotdata(filename_emu_2f_lowres,0,0, ind[0])
trace = Nee[0] + Nxx[0]
tex,Nex = plotdata(filename_emu_2f_lowres,0,1, ind[0])
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee/trace, 'k--', alpha=0.5)
axes[1].semilogy(t-tmax, Nex/trace, 'k--', alpha=0.5)

#should be same trace for sim
t,Nee = plotdata(filename_emu_2f,0,0, ind[0])
tex,Nex = plotdata(filename_emu_2f,0,1, ind[0])
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee/trace, 'k--', label=r'${\rm {\tt Emu}}$')
axes[1].semilogy(t-tmax, Nex/trace, 'k--', label=r'${\rm {\tt Emu}}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('Emu (2F)')
print('N_ee:', Nee[a_time], 'N_ex:', Nex[np.argmax(Nex)])


t,Nxx = plotdata(filename_bang_res2,1,1, ind[2])
t,Nee = plotdata(filename_bang_res2,0,0, ind[2])
tex,Nex = plotdata(filename_bang_res2,0,1, ind[2])
trace = Nee[0] + Nxx[0]
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee/trace, 'r-', alpha=0.25)
axes[1].semilogy(t-tmax, Nex/trace, 'r-', alpha=0.25)

#should be same trace for res1 and sim
t,Nee = plotdata(filename_bang_res1,0,0, ind[2])
tex,Nex = plotdata(filename_bang_res1,0,1, ind[2])
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee/trace, 'r-', alpha=0.5)
axes[1].semilogy(t-tmax, Nex/trace, 'r-', alpha=0.5)

t,Nee = plotdata(filename_bang,0,0, ind[2])
tex,Nex = plotdata(filename_bang,0,1, ind[2])
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, Nee/trace, 'r-', label=r'${\rm{\tt FLASH}}$')
axes[1].semilogy(t-tmax, Nex/trace, 'r-', label=r'${\rm{\tt FLASH}}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('FLASH')
print('N_ee:', Nee[a_time], 'N_ex:', Nex[np.argmax(Nex)])


#axes[0].legend(loc=(0.43,0.6), frameon=False)
axes[0].legend(loc='upper right', frameon=False)
plt.savefig(savedir + "NSM1_Nee_Nex_normTr_res_emu_comp.pdf", bbox_inches="tight")
