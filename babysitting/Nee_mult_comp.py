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


fig, ax = plt.subplots(1,1, figsize=(6,5))

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
filenames = ["sim0.2/reduced_data.h5", "sim1.0/reduced_data.h5"]
labels = [r"$\alpha = 0.2$", r"$\alpha = 1.0$"]
tmax_ind = [82, 37]
y_limits = (0.2, 1.05)
ax_title = r"${\rm Beam\,\,Tests}$"

##comp of perturbations in NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/od_pert/t2/xy_large/sim/reduced_data.h5"]
#labels = [r"$\delta N_{ii}$", r"$\delta N_{jk}$"]
#x_limits = (-1.3, 0.2) #ns

##comp of resolutions across NSM2.5
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/res_a/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t3/xy_large/res_a/reduced_data.h5"]
#labels = [r'$N_{gp}=16^2\times256;\,L=24.5\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times128;\,L=12.3\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times256;\,L=12.3\,{\rm cm}$']
#x_limits = (-1.0, 1.5)

lstyle = ['-', '--', '-.', ':']
lcolor = ['r', 'b', 'g', 'k', 'm']

for i,filename in enumerate(filenames):
    t,Nee = plotdata(filename,0,0)
    tex,Nex = plotdata(filename,0,1)
    txx,Nxx = plotdata(filename,1,1)
    n_2F = Nee[0] + Nxx[0]
    if tmax_ind[i] == -1:
        tmax = t[np.argmax(Nex)]
    else:
        tmax = t[tmax_ind[i]]
    #tmax = 0.0
    style_ind = i % len(lstyle)
    color_ind = i % len(lcolor)
    ax.plot(t-tmax, Nee * n_2F, linestyle=lstyle[style_ind], color=lcolor[color_ind], label=labels[i])
    #ax.plot(t-tmax, Nee * n_2F, linestyle=lstyle[style_ind], color=lcolor[color_ind])

#fig.text(0.5, 0.72, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
#fig.text(0.5, 0.67, r'$N_{{gp}}={}^3$'.format(n_grid))
#fig.text(0.55, 0.7, r'$\delta m^2 = 1\,{\rm eV}^2$')
#fig.text(0.55, 0.65, r'$\theta_{12}=0.5$')


##############
# formatting #
##############
ax.set_xlabel(r'$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$')
for i in range(2):
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
if "x_limits" in locals():
    ax.set_xlim(x_limits)
if "y_limits" in locals():
    ax.set_ylim(y_limits)
if "ax_title" in locals():
    ax.set_title(ax_title)
ax.set_ylabel(r'$\langle N_{ee}\rangle/{\rm Tr}[N]$')

ax.legend(loc=(0.0125, 0.0125), frameon=False, fontsize=20)
plt.savefig("./comp_res/Nee_mult_comp.pdf", bbox_inches="tight")
