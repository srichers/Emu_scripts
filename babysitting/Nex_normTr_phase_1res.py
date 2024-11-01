# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#neutrinos == 0; anti-neutrinos == 1
nu_type = 0

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
def plotdata(filename,nutype,a,b):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    if nu_type == 1:
        N=np.array(avgData["Nbar_avg_mag"])[:,a,b]
    else:
        N=np.array(avgData["N_avg_mag"])[:,a,b]
    avgData.close()
    return t, N

def plotdata_phase(filename,nutype):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    if nu_type == 1:
        N=np.array(avgData["Nbar_avg_phase"])[:]
    else:
        N=np.array(avgData["N_avg_phase"])[:]
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
axes[1].set_xlabel(r'$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$')
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
axes[0].set_xlim(-1.5, 0.1)
#axes[0].set_xlim(-75.0, 10.0)
#axes[0].set_xlim(-0.5, 1.0)
#axes[0].set_xlim(-0.7, 0.5)
#axes[0].set_xlim(-14.0, -10.0)
if nu_type == 1:
    axes[0].set_ylabel(r'$\langle|\overline{N}_{ex}|\rangle/{\rm Tr}[\overline{N}]$')
    axes[1].set_ylabel(r'$\langle \arg[\overline{N}_{ex}]\rangle$')
else:
    axes[0].set_ylabel(r'$\langle|N_{ex}|\rangle/{\rm Tr}[N]$')
    axes[1].set_ylabel(r'$\langle \arg[N_{ex}]\rangle$')

#axes[1].set_ylim(1.e-6,1.0)

#############
# plot data #
#############
filename_bang = "reduced_data.h5"
#filename_bang = "reduced_data_variance.h5"
#filename_bang = "reduced_data_0_cell_0_0_0.h5"
#filename_bang = "reduced0D_selection.h5"

t,Nee = plotdata(filename_bang,nu_type,0,0)
tex,Nex = plotdata(filename_bang,nu_type,0,1)
tmax = t[np.argmax(Nex)]
print('tmax = ', tmax)
axes[0].semilogy(t-tmax, Nex, 'r-', label='2f')

filename_phase = "reduced_data_phase.h5"

tphi, Nphi = plotdata_phase(filename_phase,nu_type)
axes[1].plot(tphi-tmax, Nphi, 'r-', label='2f')

#fig.text(0.5, 0.82, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
#fig.text(0.5, 0.77, r'$N_{{gp}}={}^3$'.format(n_grid))
#fig.text(0.5, 0.77, r'$N_{{gp}}={}$'.format(n_grid))
#fig.text(0.5, 0.77, r'$N_{{gp}}={}$'.format(n_grid))
#fig.text(0.5, 0.57, r'$\delta m^2=7.53\times10^{-5}\,{\rm eV}^2$')
#fig.text(0.5, 0.52, r'$\theta=0.587$')

if nu_type == 1:
    nu_type_str = 'bnu'
else:
    nu_type_str = 'nu'

#axes[0].legend(loc=(0.43,0.6), frameon=False)
namestr = "Nex_normTr_phase_1res_" + nu_type_str + ".pdf"
plt.savefig(namestr, bbox_inches="tight")
