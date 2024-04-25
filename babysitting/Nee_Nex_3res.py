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

h5name = "reduced_data.h5"
alpha_res = [1.0, 0.5, 0.25]
simres = ['sim/', 'res_a/', 'res_b/']

#fid:
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/"
#labels = [r'$N_{gp}=128^3;\,L=8.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=4.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=8.0\,{\rm cm}$']
#x_limits = (-0.2, 0.2)

#90d
#filename_1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/sim1/reduced_data.h5"
#filename_2 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/res_a1/reduced_data.h5"
#filename_3 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/res_b1/reduced_data.h5"
#label_1 = r'$N_{gp}=128^3;\,L=8.0\,{\rm cm}$'
#label_2 = r'$N_{gp}=64^3;\,L=4.0\,{\rm cm}$'
#label_3 = r'$N_{gp}=64^3;\,L=8.0\,{\rm cm}$'

#2_3:
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/"
#labels = [r'$N_{gp}=128^3;\,L=32.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=16.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=32.0\,{\rm cm}$']

#NSM_1/t2:
#filename_1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/sim2/reduced_data.h5"
#filename_2 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/res_a2/reduced_data.h5"
#filename_3 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/res_b2/reduced_data.h5"
#label_1 = r'$N_{gp}=128^3;\,L=7.87\,{\rm cm}$'
#label_2 = r'$N_{gp}=64^3;\,L=3.93\,{\rm cm}$'
#label_3 = r'$N_{gp}=64^3;\,L=7.87\,{\rm cm}$'

#NSM_1/t3:
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/"
#labels = [r'$N_{gp}=128^3;\,L=7.87\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=3.93\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=7.87\,{\rm cm}$']
#x_limits = (-0.5, 1.5)

#NSM_3:
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/"
#labels = [r'$N_{gp}=128^3;\,L=5.80\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=2.90\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=5.80\,{\rm cm}$']

#NSM_1/t3:
basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t1/"
labels = [r'$N_{gp}=128^3;\,L=24.5\,{\rm cm}$', \
    r'$N_{gp}=64^3;\,L=12.3\,{\rm cm}$', \
    r'$N_{gp}=64^3;\,L=24.5\,{\rm cm}$']
x_limits = (-1.0, 1.5)
#from LSA:
est_imo = 1.39e10 #s^{-1}
est_imo_ns = est_imo/1.e+9 #ns^{-1}
Nex_base = 10.0**(-3.5)
scale_fact = 1.e+1
ind_offset = 4

for i in range(3):
    filename = basedir + simres[i] + h5name
    t,Nee = plotdata(filename,0,0)
    tex,Nex = plotdata(filename,0,1)
    txx,Nxx = plotdata(filename,1,1)
    n_2F = Nee[0] + Nxx[0]
    tmax = t[np.argmax(Nex)]
    axes[0].plot(t-tmax, Nee * n_2F, 'r-', alpha=alpha_res[i], label=labels[i])
    if i == 0:
        n_2F_eq = n_2F/2.0
        axes[0].axhline(n_2F_eq, color="green")
        if "est_imo_ns" in locals():
            ind1 = np.argmin(np.abs(np.log(Nex[1:]/Nex_base)))
            ind2 = ind1 + ind_offset
            t_line = [t[ind1], t[ind2]]
            N_line = [scale_fact*Nex[ind1], scale_fact*Nex[ind1]*np.exp(est_imo_ns*(t[ind2] - t[ind1]))]
            axes[1].semilogy(t_line-tmax, N_line, color='orange')
    axes[1].semilogy(t-tmax, Nex * n_2F, 'r-', alpha=alpha_res[i],  label=labels[i])


##############
# formatting #
##############
axes[1].set_xlabel(r'$t-t_{\rm sat}\,({\rm s})$')
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
axes[0].set_xlim(x_limits)
axes[0].set_ylabel(r'$\langle N_{ee}\rangle/{\rm Tr}[N]$')
axes[1].set_ylabel(r'$\langle|N_{ex}|\rangle/{\rm Tr}[N]$')

#axes[0].legend(loc=(0.43,0.1), frameon=False)
#axes[0].legend(loc='best', frameon=False)
axes[1].legend(loc='best', frameon=False)
savename = basedir + "comp_res/Nee_Nex_3res.pdf"
plt.savefig(savename, bbox_inches="tight")
