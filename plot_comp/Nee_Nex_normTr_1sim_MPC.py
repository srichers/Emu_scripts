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

#############
# plot data #
#############

#fid
#emu_test = "Fiducial"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_3F/reduced_data.h5"
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/sim/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#savename =  "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
#xlim = (-0.5, 0.2)

#90d
#emu_test = "90Degree"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_3F/reduced_data.h5"
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/sim1/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#savename =  "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
#xlim = (-0.5, 0.1)

#2_3
#emu_test = "TwoThirds"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_3F/reduced_data.h5"
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/sim/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#ind[0] = 1
#savename =  "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
#xlim = (-1.5, 0.2)

#NSM_3:
emu_test = "NSM_3"
filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_3/32dir/merger_2F/plt_reduced_data.h5"
filename_emu_3f = None
filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t1/sim/reduced_data.h5"
ind = np.zeros([3], dtype=np.int8)
ind[0] = 1
savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t1/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
xlim = (-0.5, 0.5)

#ind = np.zeros([3], dtype=np.int8)
#ind[0] = 1

t,Nee = plotdata(filename_emu_2f,0,0,ind[0])
t,Nxx = plotdata(filename_emu_2f,1,1,ind[0])
t_ex,N_ex = plotdata(filename_emu_2f,0,1,ind[0])
#special code for renormalizing emu_2F:
Ntr_2f = Nee[0] + Nxx[0]
Nee = Nee/Ntr_2f
N_ex = N_ex/Ntr_2f
tmax = t[np.argmax(N_ex)]
#tdec = tmax + delta_max_dec #decoherence time after sat.
#tdec_ind = np.argmin(abs(t-tdec)) #index where tdec falls in t
axes[0].plot(t-tmax, Nee, 'k-', label=r'${\rm {\tt Emu}\,\,(2f)}$')
axes[1].semilogy(t-tmax, N_ex, 'k-', label=r'${\rm {\tt Emu}\,\,(2f)}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('Emu (2F)')
print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])


if filename_emu_3f != None:
    t,Nee = plotdata(filename_emu_3f,0,0,ind[1])
    t,Nmm = plotdata(filename_emu_3f,1,1,ind[1])
    t,Ntt = plotdata(filename_emu_3f,2,2,ind[1])
    Ntr_3f = Nee[0] + Nmm[0] + Ntt[0]
    t_ex,N_ex = plotdata(filename_emu_3f,0,1,ind[1])
    tmax = t[np.argmax(N_ex)]
    #special code for excising a single point
    if emu_test == "Fiducial":
        bad_ind = 152
        t = np.concatenate((t[:bad_ind-1], t[bad_ind+1:]))
        Nee = np.concatenate((Nee[:bad_ind-1], Nee[bad_ind+1:]))
        t_ex = np.concatenate((t_ex[:bad_ind-1], t_ex[bad_ind+1:]))
        N_ex = np.concatenate((N_ex[:bad_ind-1], N_ex[bad_ind+1:]))
    axes[0].plot(t-tmax, Nee, 'k--', label=r'${\rm {\tt Emu}\,\,(3f)}$')
    axes[1].semilogy(t-tmax, N_ex, 'k--', label=r'${\rm {\tt Emu}\,\,(3f)}$')
    a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
    print('Emu (3F)')
    print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])


t,Nee = plotdata(filename_bang,0,0,ind[2])
t_ex,N_ex = plotdata(filename_bang,0,1,ind[2])
tmax = t[np.argmax(N_ex)]
#tdec = tmax + delta_max_dec #decoherence time after sat.
#tdec_ind = np.argmin(abs(t-tdec)) #index where tdec falls in t
axes[0].plot(t-tmax, Nee, 'r-', label=r'${\rm {\tt FLASH}\,\,(2f)}$')
axes[1].semilogy(t-tmax, N_ex, 'r-', label=r'${\rm {\tt FLASH}\,\,(2f)}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('FLASH')
print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])

axes[0].axhline(1.0/2.0, color="green")
axes[0].axhline(1.0/3.0, color="green", linestyle='--')

##############
# formatting #
##############
axes[1].set_xlabel(r"$t-t_{\rm max}\,(10^{-9}\,\mathrm{s})$")
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
axes[0].set_xlim(xlim[0], xlim[1])
axes[0].set_ylabel(r"$\langle N_{ee}\rangle/{\rm Tr}[N]$")
axes[1].set_ylabel(r"$\langle |N_{ex}|\rangle/{\rm Tr}[N]$")

axes[1].legend(loc='lower right', fontsize=12, frameon=False)
plt.savefig(savename, bbox_inches="tight")
