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

##fid
#emu_test = "Fiducial"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_3F/reduced_data.h5"
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/t4/xy_large/sim/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#savename =  "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/t4/xy_large/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
#xlim = (-0.5, 2.0)

#90d
#emu_test = "90Degree"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_3F/reduced_data.h5"
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/t2/xy_large/sim/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#savename =  "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/t2/xy_large/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
#xlim = (-0.5, 2.0)

#2_3
#emu_test = "TwoThirds"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_2F/reduced_data.h5"
#filename_emu_3f = "/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_3F/reduced_data.h5"
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/t2/xy_large/sim/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#ind[0] = 1
#savename =  "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/t2/xy_large/comp_emu/Nee_Nex_normTr_1sim_MPC.pdf"
#xlim = (-1.5, 2.0)

#NSM_3/t1:
#emu_test = "NSM_3"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_3/32dir/merger_2F/plt_reduced_data.h5"
#filename_emu_3f = None
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/xy_large/sim/long_t/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#ind[0] = 1
#tmax_ind = -1
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/xy_large/comp_emu/Nee_Nex_emu_comp.pdf"
#xlim = (-0.6, 0.5)
#est_imo = 2.5e10 #s^{-1}
#Nex_pow = -2.7
#scale_fact = 5.0
#ind_offset = 15
#ax_title = r'${\rm NSM}3$'

##NSM_3/t3 xy_large:
#emu_test = "NSM_3"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_3/32dir/merger_2F/plt_reduced_data.h5"
#filename_emu_3f = None
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/sim_xy_large/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#ind[0] = 1
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/comp_emu/xy_large/Nee_Nex_normTr_1sim_MPC_long_t.pdf"
#xlim = (-0.5, 2.0)

##Beam/rand/changing_N_nuebar/sim1.0
#emu_test = "Beam_1.0"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series/neebar_1.0/reduced0D_selection.h5"
#filename_emu_3f = None
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/sim1.0/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/sim1.0/comp_emu/Nee_Nex_normTr_1sim.pdf"
#xlim = (-0.1, 0.25)
##need to set tmax to below for filename_emu_2f to get good comparison with FLASH:
##tmax = t[177]

##NSM_2/clos3/t4/xy_large:
emu_test = "NSM_2"
filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_2/32dir/merger_2F/plt_reduced_data.h5"
filename_emu_3f = None
filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2/MPC/clos3/d_pert/t4/xy_large/sim/reduced_data.h5"
ind = np.zeros([3], dtype=np.int8)
ind[0] = 1
tmax_ind = 235
savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2/MPC/clos3/d_pert/t4/xy_large/comp_emu/Nee_Nex_1sim_emu_comp.pdf"
xlim = (-1.0, 0.2)
#from LSA:
est_imo = 1.26e10 #s^{-1}
Nex_pow = -3.0
scale_fact = 3.0
ind1 = 205
ind_offset = 13
ax_title = r'${\rm NSM}2$'
tmax_ind = 230

#NSM_4/t2/xy_large, and Emu/NSM_4/1res/correct_Nxx/NSM_4_matchEvan*/plt**:
#emu_test = "NSM_4"
#filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_4/1res/correct_Nxx/NSM4_matchevan_correctNxx_long_diagonalpert/plt_reduced_data.h5"
#filename_emu_3f = None
#filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_4/d_pert/t2/xy_large/sim/reduced_data.h5"
#ind = np.zeros([3], dtype=np.int8)
#ind[0] = 1
#tmax_ind = -1
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_4/d_pert/t2/xy_large/comp_emu/plt/Nee_Nex_1sim_emu_comp.pdf"
#xlim = (-4.0, 4.0)
###from LSA:
##est_imo = 1.39e10 #s^{-1}
##Nex_pow = -2.2
##scale_fact = 10.0
##ind1 = 60
##ind_offset = 8
#ax_title = r'${\rm NSM}4$'

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
#one-time only for Beam/sim1.0:
#tmax = t[177]
#tdec = tmax + delta_max_dec #decoherence time after sat.
#tdec_ind = np.argmin(abs(t-tdec)) #index where tdec falls in t
axes[0].plot(t-tmax, Nee, 'k--', label=r'${\rm {\tt Emu}}$')
axes[1].semilogy(t-tmax, N_ex, 'k--', label=r'${\rm {\tt Emu}}$')
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
    #axes[0].axhline(1.0/3.0, color="green", linestyle='--')
    axes[1].semilogy(t-tmax, N_ex, 'k--', label=r'${\rm {\tt Emu}\,\,(3f)}$')
    a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
    print('Emu (3F)')
    print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])


t,Nee = plotdata(filename_bang,0,0,ind[2])
t_ex,N_ex = plotdata(filename_bang,0,1,ind[2])
if tmax_ind == -1:
    tmax = t[np.argmax(N_ex)]
else:
    tmax = t[tmax_ind]
#tdec = tmax + delta_max_dec #decoherence time after sat.
#tdec_ind = np.argmin(abs(t-tdec)) #index where tdec falls in t
axes[0].plot(t-tmax, Nee, 'r-', label=r'${\rm {\tt FLASH}}$')
axes[1].semilogy(t-tmax, N_ex, 'r-', label=r'${\rm {\tt FLASH}}$')
a_time = min(range(len(t)), key=lambda i: abs(t[i]-2.0))
print('FLASH')
print('N_ee:', Nee[a_time], 'N_ex:', N_ex[np.argmax(N_ex)])

if "est_imo" in locals():
    est_imo_ns = est_imo/1.e+9 #ns^{-1}
    Nex_base = 10.0**(Nex_pow)
    if "ind1" not in locals():
        ind1 = np.argmin(np.abs(np.log(N_ex[1:]/Nex_base)))
    ind2 = ind1 + ind_offset
    t_line = [t[ind1], t[ind2]]
    N_line = [scale_fact*N_ex[ind1], scale_fact*N_ex[ind1]*np.exp(est_imo_ns*(t[ind2] - t[ind1]))]
    axes[1].semilogy(t_line-tmax, N_line, color='orange', label=None)

axes[0].axhline(1.0/2.0, color="green", label=None)

##############
# formatting #
##############
axes[1].set_xlabel(r"$t-t_{\rm sat}\,(10^{-9}\,\mathrm{s})$")
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
if "xlim" in locals():
    axes[0].set_xlim(xlim[0], xlim[1])
if "ax_title" in locals():
    axes[0].set_title(ax_title)
axes[0].set_ylabel(r"$\langle N_{ee}\rangle/\langle{\rm Tr}[N]\rangle$")
axes[1].set_ylabel(r"$\langle |N_{ex}|\rangle/\langle{\rm Tr}[N]\rangle$")

axes[1].legend(loc='lower right', frameon=False)
plt.savefig(savename, bbox_inches="tight")
