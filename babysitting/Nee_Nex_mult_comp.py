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
    
t_strs = ["t", "t(s)"]
N_strs = ["N_avg_mag", "N_avg_mag(1|ccm)"]

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename,a,b,tind):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData[t_strs[tind]])*1e9
    N=np.array(avgData[N_strs[tind]])[:,a,b]
    #N=np.array(avgData["Nbar_avg_mag"])[:,a,b]
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
#filename_1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/sim1/reduced_data.h5"
#filenames = ["/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert4/reduced_data.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert3/reduced_data.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert1/reduced_data.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert2/reduced_data.h5"]
#labels = [r'$\delta m^2=7.53\times10^{-5}\,{\rm eV}^2$', \
#        r'$\delta m^2=0.1\,{\rm eV}^2$', \
#        r'$\delta m^2=1\,{\rm eV}^2$', \
#        r'$\delta m^2=10\,{\rm eV}^2$']
#filenames = ["./sim1/reduced_data.h5", \
#    "./sim1_od/reduced_data.h5"]
#labels = [r'$H_V+H_\nu$', \
#        r'$H_\nu$']
#box_length = 62.83185307179586
#n_grid = 128

#filenames = ["/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/sim_clos2/reduced_data.h5", \
#    "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/sim/reduced_data.h5"]
#labels = [r'${\rm New}$', \
#        r'${\rm old\,\,(FT)}$']

#Beam in FFI_1D:
#filenames = ["sim" + str(i+1) + "/reduced_data.h5" for i in range(4)]
#labels = [r"$N_{{gp}} = {}$".format(1024/2**(i)) for i in range(4)]

#Beam in FFI_1D/changing_N_nuebar:
#filenames = ["sim" + str((i+1)*0.2)[0:3] + "/reduced_data.h5" for i in range(5)]
#labels = [r"$\overline{{N}}_{{ee}}/N_{{ee}} = {:.1f}$".format((i+1)*0.2) for i in range(5)]
#filenames = ["sim0.2/reduced_data.h5", "sim1.0/reduced_data.h5"]
#labels = [r"$\overline{{N}}_{{ee}}/N_{{ee}} = 0.2$", r"$\overline{{N}}_{{ee}}/N_{{ee}} = 1.0$"]

##comp of perturbations in NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/od_pert/t2/xy_large/sim/reduced_data.h5"]
#labels = [r"$\delta N_{ii}$", r"$\delta N_{jk}$"]
#xlimits = (-1.3, 0.2) #ns

##comp of resolutions across NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/res_a/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t3/xy_large/res_a/reduced_data.h5"]
#labels = [r'$N_{gp}=16^2\times256;\,L=24.5\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times128;\,L=12.3\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times256;\,L=12.3\,{\rm cm}$']
#x_limits = (-1.0, 1.5)
#y_Nex_limits = (1.e-3,1.e-1)

#Beam in FFI_1D/chaos_study:
#filenames = ["sim" + str((i+1)) + "/reduced_data.h5" for i in range(5)]
#labels = [r"$\overline{{N}}_{{ee}}/N_{{ee}} = {:.1f}$".format((i+1)*0.2) for i in range(5)]

##comp of res_a across NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t1/res_a/reduced_data.h5", \
#             "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/res_a/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t3/xy_large/res_a/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t4/xy_large/res_a/reduced_data.h5"]
#labels = [r'$N_{gp}=16^2\times64$', r'$N_{gp}=16^2\times128$', \
#    r'$N_{gp}=16^2\times256$', r'$N_{gp}=16^2\times512$']
#x_limits = (-1.0, 1.5)
#y_Nex_limits = (1.e-4,1.e-1)
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/comp_t/Nee_Nex_mult_comp.pdf"

##comp of res_a across NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t3/xy_large/res_a/reduced_data.h5", \
#    "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/sin_pert/xy_large/sim8/reduced_data.h5", \
#    "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_2.5/1res/correct_Nxx/NSM2.5_matchevan_correctNxx_long_diagonalpert/plt_reduced_data.h5"]
#labels = [r'${\rm Rand-Diag}\,(\texttt{FLASH})$', \
#    r'${\rm Sin-off-Diag}\,(\texttt{FLASH})$', \
#    r'${\rm Rand-Diag}\,(\texttt{Emu})$']
#tind = np.zeros(len(filenames), dtype=np.int64)
#tind[2] = 1
#x_limits = (-1.0, 1.5)
#y_Nex_limits = (1.e-4,1.e-1)
#savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/sin_pert/xy_large/comp/Nee_Nex_mult_comp.pdf"

#comp of 2/3 from t3 and t5
filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/t3/xy_large/sim/reduced_data.h5", \
    "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/t5/xy_large/res_b/reduced_data.h5"]
labels = [r'${\rm sim}\,\delta N_{cc}=10^{-6}$', \
    r'${\rm res\_b}\,\delta N_{cc}=10^{-4}$']
tind = np.zeros(len(filenames), dtype=np.int64)
#x_limits = (-1.0, 1.5)
#y_Nex_limits = (1.e-4,1.e-1)
savename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/comp_t/Nee_Nex_mult_comp.pdf"

#lstyle = ['-', '--', '-.', ':']
#lcolor = ['r', 'b', 'g', 'k', 'm']
lstyle = ['-', '--', '-']
lcolor = ['r', 'b', 'k']

for i,filename in enumerate(filenames):
    t,Nee = plotdata(filename,0,0,tind[i])
    tex,Nex = plotdata(filename,0,1,tind[i])
    txx,Nxx = plotdata(filename,1,1,tind[i])
    tr_N = Nee[0] + Nxx[0]
    Nee = Nee/tr_N
    Nxx = Nxx/tr_N
    Nex = Nex/tr_N
    tmax = t[np.argmax(Nex)]
    #tmax = 0.0
    style_ind = i % len(lstyle)
    color_ind = i % len(lcolor)
    #axes[0].plot(t-tmax, Nee, linestyle=lstyle[style_ind], color=lcolor[color_ind], label=labels[i])
    axes[0].plot(t-tmax, Nee, linestyle=lstyle[style_ind], color=lcolor[color_ind])
    axes[1].semilogy(t-tmax, Nex, label=labels[i], linestyle=lstyle[style_ind], color=lcolor[color_ind])

#fig.text(0.5, 0.72, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
#fig.text(0.5, 0.67, r'$N_{{gp}}={}^3$'.format(n_grid))
#fig.text(0.55, 0.7, r'$\delta m^2 = 1\,{\rm eV}^2$')
#fig.text(0.55, 0.65, r'$\theta_{12}=0.5$')


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
#axes[0].set_xlim(-0.5, 0.5)
#axes[0].set_xlim(-0.25, 0.75)
if "x_limits" in locals():
    axes[0].set_xlim(x_limits)
#else:
#    axes[0].set_xlim(-0.4, 0.8)
if "y_Nex_limits" in locals():
    axes[1].set_ylim(y_Nex_limits)
axes[0].set_ylabel(r'$\langle N_{ee}\rangle/{\rm Tr}[N]$')
axes[1].set_ylabel(r'$\langle|N_{ex}|\rangle/{\rm Tr}[N]$')
#axes[0].set_ylabel(r'$\langle \overline{N}_{ee}\rangle/{\rm Tr}[\overline{N}]$')
#axes[1].set_ylabel(r'$\langle|\overline{N}_{ex}|\rangle/{\rm Tr}[\overline{N}]$')

#axes[0].legend(loc=(0.43,0.1), frameon=False)
axes[1].legend(loc='best', frameon=False)
#plt.savefig("./comp_res/Nee_Nex_2_comp.pdf", bbox_inches="tight")
#plt.savefig("./comp_res/Nee_Nex_mult_comp_bnu.pdf", bbox_inches="tight")
plt.savefig(savename, bbox_inches="tight")
