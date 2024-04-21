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

#90d MPC names:
dirs = ["./sim1/", "res_a1/", "res_b1/"]
labels = [r"$N_{{gp}} = {}^3;\,L={}\,{{\rm cm}}$".format(128, 8), \
    r"$N_{{gp}} = {}^3;\,L={}\,{{\rm cm}}$".format(64, 4), \
    r"$N_{{gp}} = {}^3;\,L={}\,{{\rm cm}}$".format(64, 8)]

lstyle = ['-', '--', '-.', ':']
lcolor = ['r', 'b', 'g', 'k', 'm']

for i,dirname in enumerate(dirs):
    filename = dirname + "reduced_data.h5"
    t,Nee = plotdata(filename,0,0)
    tex,Nex = plotdata(filename,0,1)
    txx,Nxx = plotdata(filename,1,1)
    n_2F = Nee[0] + Nxx[0]
    tmax = t[np.argmax(Nex)]
    #tmax = 0.0
    style_ind = i % len(lstyle)
    color_ind = i % len(lcolor)
    #axes[0].plot(t-tmax, Nee * n_2F, linestyle=lstyle[style_ind], color=lcolor[color_ind], label=labels[i])
    axes[0].plot(t-tmax, Nee * n_2F, linestyle=lstyle[style_ind], color=lcolor[color_ind])
    n_2F_eq = n_2F/2.0
    #if i == 0:
    #    axes[0].axhline(n_2F_eq, color="c")
    #axes[1].semilogy(t-tmax, Nex * n_2F, linestyle=lstyle[i], color=lcolor[i])
    axes[1].semilogy(t-tmax, Nex * n_2F, label=labels[i], linestyle=lstyle[i], color=lcolor[i])

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
axes[0].set_xlim(-0.4, 0.8)
#axes[0].set_xlim(-0.25, 0.75)
axes[0].set_ylabel(r'$\langle N_{ee}\rangle/{\rm Tr}[N]$')
axes[1].set_ylabel(r'$\langle|N_{ex}|\rangle/{\rm Tr}[N]$')

#axes[0].legend(loc=(0.43,0.1), frameon=False)
axes[1].legend(loc='best', frameon=False)
#plt.savefig("Nee_Nex_mult_comp.pdf", bbox_inches="tight")
plt.savefig("./comp/Nee_Nex_mult_comp.pdf", bbox_inches="tight")
