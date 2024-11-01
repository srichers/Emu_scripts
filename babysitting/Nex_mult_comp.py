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

    
t_str = ["t", "t(s)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]

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


fig, axes = plt.subplots(1,1, figsize=(6,5))
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

##Beam in FFI_1D/changing_N_nuebar:
#filenames = ["sim" + str((i+1)*0.2)[0:3] + "/reduced_data.h5" for i in range(5)]
#labels = [r"$\overline{{N}}_{{ee}}/N_{{ee}} = {:.1f}$".format((i+1)*0.2) for i in range(5)]

#comp of perturbations in NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/od_pert/t2/xy_large/sim/reduced_data.h5"]
#labels = [r"$\delta N_{cc}$", r"$\delta N_{ab}$"]
#xlimits = (-1.3, 0.2) #ns

#comp of methods in NSM1
filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/sim/reduced_data.h5", \
        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/sim/reduced_data.h5", \
        "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/merger_2F/plt_reduced_data.h5"]
ind = np.zeros([3], dtype=np.int8)
ind[2] = 1
labels = [r"${\rm {\tt FLASH}}$", r"${\rm {\tt FLASH}}\,\,(ri)$", r"${\rm {\tt Emu}}$"]
xlimits = (-0.3, 0.3) #ns

lstyle = ['-', '-.', '--', ':']
lcolor = ['r', 'b', 'k', 'g', 'm']

for i,filename in enumerate(filenames):
    t,Nee = plotdata(filename,0,0,ind[i])
    tex,Nex = plotdata(filename,0,1,ind[i])
    txx,Nxx = plotdata(filename,1,1,ind[i])
    n_2F = Nee[0] + Nxx[0]
    tmax = t[np.argmax(Nex)]
    #tmax = 0.0
    style_ind = i % len(lstyle)
    color_ind = i % len(lcolor)
    n_2F_eq = n_2F/2.0
    axes.semilogy(t-tmax, Nex/n_2F, label=labels[i], linestyle=lstyle[style_ind], color=lcolor[color_ind])


##############
# formatting #
##############
axes.set_xlabel(r'$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$')
axes.tick_params(axis='both', which='both', direction='in', right=True,top=True)
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.yaxis.set_minor_locator(AutoMinorLocator())
axes.minorticks_on()
if "xlimits" in locals():
    axes.set_xlim(xlimits)
else:
    axes.set_xlim(-0.4, 0.8)
axes.set_ylabel(r'$\langle|N_{ex}|\rangle/\langle{\rm Tr}[N]\rangle$')

axes.legend(loc='best', frameon=False)
#plt.savefig("Nex_mult_comp.pdf", bbox_inches="tight")
plt.savefig("./method_comp/Nex_mult_comp.pdf", bbox_inches="tight")
