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

clight = 29979245800.0
    
n_nue0 = 1.421954234999705e+33     # 1/ccm
n_nux0 = 1.9645407875568215e+33/4. # 1/ccm, each flavor
n_tot = n_nue0 + 2.*n_nux0
n_2F = n_nue0 + n_nux0
mfact = 1.e-32
mfactstr = r'$10^{-32}\times$'

flash_ffact_conv = 50.0/(4.0*np.pi)

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename_avg, d, a, b):
    
    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData["t"])*1e9
    Nexavg=np.array(avgData["N_avg_mag"][:,0,1])

    # make time relative to tmax
    itmax = np.argmax(Nexavg)
    t = t-t[itmax]

    Num=np.array(avgData["N_avg_mag"][:,a,b])

    N=np.array(avgData["F_avg_mag"])[:,d,a,b]
    avgData.close()

    return t, N


################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)
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


fig, axes = plt.subplots(3,1, figsize=(6,12), sharex=True)
plt.subplots_adjust(hspace=0)

##############
# formatting #
##############
axes[2].set_xlabel(r"$t-t_{\rm max}\,(10^{-9}\,\mathrm{s})$")
for i in range(3):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
    #axex[i].grid(which='both')
axes[0].set_xlim(-1.0, 4.0)
axes[0].set_ylabel(r"$\langle |F^{(x)}_{ee}|\rangle\,({\rm cm}^{-3})$")
axes[1].set_ylabel(r"$\langle |F^{(y)}_{ee}|\rangle\,({\rm cm}^{-3})$")
axes[2].set_ylabel(r"$\langle |F^{(z)}_{ee}|\rangle\,({\rm cm}^{-3})$")
#axes[0].set_ylim(0.9*mfact*n_nux0, 1.1*mfact*n_nue0)


#############
# plot data #
#############
tplot = -0.1e-9
basedirs = ["/global/project/projectdirs/m3761/Evan/",
            "/global/project/projectdirs/m3761/Evan/",
            "/global/project/projectdirs/m3761/FLASH/FFI_3D/"]
simlist = ["merger_2F/", "merger_3F/", "NSM/"]

filename_emu_2f = basedirs[0]+simlist[0]+"reduced_data_unitful.h5"
filename_emu_3f = basedirs[1]+simlist[1]+"reduced_data_unitful.h5"
filename_bang   = basedirs[2]+simlist[2]+"sim1/reduced_data_NSM_sim_hdf5_chk.h5"

for d in range(3):
    t,F = plotdata(filename_emu_2f,d,0,0)
    axes[d].semilogy(t, F, 'k-', label=r'${\rm Emu\,\,(2f)}$')
    t,F = plotdata(filename_emu_3f,d,0,0)
    axes[d].semilogy(t, F, 'k--', label=r'${\rm Emu\,\,(3f)}$')
    t,F = plotdata(filename_bang,d,0,0)
    #need conversion factor for flash data
    axes[d].semilogy(t, F*flash_ffact_conv*n_2F, 'r-', label=r'${\rm FLASH\,\,(2f)}$')

############
# save pdf #
############
plt.savefig("Flux_3dirs_avg_ee.pdf", bbox_inches="tight")
