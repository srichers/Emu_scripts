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
#n_nue0 = 2.3293607911671233e+33    # 1/ccm
#n_nux0 = 1.5026785300973756e+33 # 1/ccm, each flavor
#NSM_3:
n_nue0 = 2.8800567085107055e+33    # 1/ccm
n_nux0 = 4.831622183948198e+32 # 1/ccm, each flavor
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
axes[0].axhline(mfact*n_2F_eq, color="silver")
#axes[0].axhline(mfact*n_tot_eq, color="green", linestyle='--')
axes[1].set_xlabel(r"$t-t_{\rm max}\,(10^{-9}\,\mathrm{s})$")
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
#axes[0].set_xlim(-1.0, 4.0)
axes[0].set_xlim(-0.5, 2.0)
axes[0].set_ylabel(mfactstr + r"$\langle N_{ee}\rangle\,({\rm cm}^{-3})$")
axes[1].set_ylabel(mfactstr + r"$\langle |N_{ex}|\rangle\,({\rm cm}^{-3})$")
axes[0].set_ylim(0.9*mfact*n_nux0, 1.1*mfact*n_nue0)

#############
# plot data #
#############
#NSM_3:
NSM_dir = 'NSM_3/'
#t1
#box_length = 5.803047682077095
#n_grid = 128
#try_dir = 't1'
#t2
#box_length = 5.803047682077095
#n_grid = 256
#try_dir = 't2'
#t3
#box_length = 2.9015238410385473 #cm
#n_grid = 128
#try_dir = 't3'
#t4
#box_length = 11.60609536415419
#n_grid = 256
#try_dir = 't4'
#t5
box_length = 11.60609536415419
n_grid = 512
try_dir = 't5'

filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/" + NSM_dir + try_dir + "/sim/reduced_data.h5"
filename_bang_res1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/" + NSM_dir + try_dir + "/res_a/reduced_data.h5"
filename_bang_res2 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/" + NSM_dir + 't2' + "/res_a/reduced_data.h5"
plot_dir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/" + NSM_dir

t,Nee = plotdata(filename_bang,0,0)
tex,Nex = plotdata(filename_bang,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'r-', label = r'$N_{{gp}}/L={}/{:.3f}\,{{\rm cm}}$'.format(n_grid,box_length))
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'r-')

t,Nee = plotdata(filename_bang_res1,0,0)
tex,Nex = plotdata(filename_bang_res1,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'g--', label = r'$N_{{gp}}/L={}/{:.3f}\,{{\rm cm}}$'.format(n_grid/2,box_length/2.0))
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'g--')

t,Nee = plotdata(filename_bang_res2,0,0)
tex,Nex = plotdata(filename_bang_res2,0,1)
tmax = t[np.argmax(Nex)]
axes[0].plot(t-tmax, mfact * Nee * n_2F, 'b-', label = r'$N_{{gp}}/L={}/{:.3f}\,{{\rm cm}}$'.format(n_grid/4,box_length/4.0))
axes[1].semilogy(t-tmax, mfact * Nex * n_2F, 'b-')


#fig.text(0.5, 0.82, r'$L_s={:.3f}\,{{\rm cm}}$'.format(box_length))
#fig.text(0.5, 0.77, r'$N_{{gp,s}}={}^3$'.format(n_grid))

axes[0].legend(loc=(0.43,0.6), fontsize=14, frameon=False)
plotfile = plot_dir + "Nee_Nex_2panels_special.pdf"
plt.savefig(plotfile, bbox_inches="tight")
