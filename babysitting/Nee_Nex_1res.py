# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#NSM_1:
#n_nue0 = 1.421954234999705e+33     # 1/ccm
#n_nux0 = 1.9645407875568215e+33/4. # 1/ccm, each flavor
#NSM_2:
n_nue0 = 2.3293607911671233e+33    # 1/ccm
n_nux0 = 1.5026785300973756e+33 # 1/ccm, each flavor
n_tot = n_nue0 + 2.*n_nux0
n_2F = n_nue0 + n_nux0
n_tot_eq = n_tot/3.0
n_2F_eq = n_2F/2.0

mfact = 1.e-33
mfactstr = r'$10^{-33}\times$'

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
#filename_bang = "reduced_data_NSM_sim.h5"
filename_bang = "reduced_data.h5"
#t1
#t2:
#sim:
#box_length = 4.132703957221158
#n_grid = 256
#res_a
#box_length = 2.066351978610579
#n_grid = 128
#t3:
#res_b:
#box_length = 8.265407914442315
#n_grid = 128
#t4:
#sim:
box_length = 16.53081582888463
n_grid = 512
#filename_bang = "reduced_data_NSM_res2.h5"

t,Nee = plotdata(filename_bang,0,0)
tex,Nex = plotdata(filename_bang,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'r-', label='2f')
axes[0].axhline(mfact*n_2F_eq, color="green")
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'r-', label='2f')

fig.text(0.5, 0.82, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
fig.text(0.5, 0.77, r'$N_{{gp}}={}^3$'.format(n_grid))


#axes[0].legend(loc=(0.43,0.6), frameon=False)
plt.savefig("Nee_Nex_1res.pdf", bbox_inches="tight")
