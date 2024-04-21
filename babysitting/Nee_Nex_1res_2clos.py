# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#2/3:
n_nue0 = 4.89e+32    # 1/ccm
n_nux0 = 0.0 # 1/ccm, each flavor

n_tot = n_nue0 + 2.*n_nux0
n_2F = n_nue0 + n_nux0
n_tot_eq = n_tot/3.0
n_2F_eq = n_2F/2.0

mfact = 1.e-32
mfactstr = r'$10^{-32}\times$'

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
axes[1].set_xlabel(r'$t\,({\rm s})$')
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
#axes[0].set_xlim(-1.0, 4.0)
axes[0].set_xlim(-0.5, 2.0)
axes[0].set_ylabel(mfactstr + r'$\langle N_{ee}\rangle\,({\rm cm}^{-3})$')
axes[1].set_ylabel(mfactstr + r'$|N_{ex}|\,({\rm cm}^{-3})$')
axes[0].set_ylim(0.9*mfact*n_nux0, 1.1*mfact*n_nue0)

#############
# plot data #
#############
filename_mec = "/global/project/projectdirs/m3761/FLASH/FFI_3D/2_3/sim1/reduced_data_nov4_test_hdf5_chk.h5"
filename_thick = "/global/project/projectdirs/m3761/FLASH/FFI_3D/2_3_clos/sim/reduced_data.h5"
#2/3:
box_length = 32.0
n_grid = 128

t_mec,Nee_mec = plotdata(filename_mec,0,0)
tex_mec,Nex_mec = plotdata(filename_mec,0,1)
tmax_mec = t_mec[np.argmax(Nex_mec)]
axes[0].plot(t_mec-tmax_mec, mfact * Nee_mec * n_2F, 'r-', label=r'${\rm MEC}$')
axes[0].axhline(mfact*n_2F_eq, color="green")
axes[1].semilogy(t_mec-tmax_mec, mfact * Nex_mec * n_2F, 'r-')

t_thick,Nee_thick = plotdata(filename_thick,0,0)
tex_thick,Nex_thick = plotdata(filename_thick,0,1)
tmax_thick = t_thick[np.argmax(Nex_thick)]
axes[0].plot(t_thick-tmax_thick, mfact * Nee_thick * n_2F, 'b-', label=r'$\chi=1/3$')
axes[1].semilogy(t_thick-tmax_thick, mfact * Nex_thick * n_2F, 'b-')

fig.text(0.5, 0.82, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
fig.text(0.5, 0.77, r'$N_{{gp}}={}^3$'.format(n_grid))


axes[0].legend(loc=(0.43,0.1), frameon=False)
plt.savefig("Nee_Nex_1res_2clos.pdf", bbox_inches="tight")
