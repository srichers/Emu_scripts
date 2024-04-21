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
    
t_str = ["t", "t(s)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


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


fig = plt.figure(figsize=(12,6))

test_list = [['Fiducial', 'fid'], ['90Degree', '90d'], ['TwoThirds', '2_3']]
test_fig_labels = [r'${\rm Fiducial}$', r'${\rm 90Degree}$', r'${\rm TwoThirds}$']

ind = np.zeros([3,3], dtype=np.int8)
#special cases
ind[0,2] = 1

test_titles = [r'${\rm Fiducial}$', r'${\rm 90Degree}$', r'${\rm TwoThirds}$']

fig, axes = plt.subplots(2,1, figsize=(6,10), sharex=True)
plt.subplots_adjust(hspace=0)

#############
# plot data #
#############
filename_emu_2f = "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_2F/reduced_data.h5"
t,N = plotdata(filename_emu_2f,0,0,0)
t_ex,N_ex = plotdata(filename_emu_2f,0,1,0)
#special code for renormalizing emu_2F:
N0 = N[0]
N = N/N0
N_ex = N_ex/N0
tmax = t[np.argmax(N_ex)]
axes[0].plot(t-tmax, N, 'k-', label=None)
axes[1].semilogy(t-tmax, N_ex, 'k-', label=r'${\rm {\tt EMU}\,\,(2f)}$')

filename_bang = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/MPC/fid/sim/reduced_data.h5"
t,N = plotdata(filename_bang,0,0,0)
t_ex,N_ex = plotdata(filename_bang,0,1,0)
tmax = t[np.argmax(N_ex)]
axes[0].plot(t-tmax, N, 'b-', label=None)
#ax.text(x=2.5, y=0.9, s=test_fig_labels[i], fontsize=12)
axes[1].set_xlabel(r"$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$")
axes[1].semilogy(t-tmax, N_ex, 'b-', label=r'${\rm MPC\,\,(2f)}$')

axes[1].legend(loc='lower right', fontsize=12, frameon=False)


##############
# formatting #
##############
for i in range(2):
    axes[i].tick_params(axis='both', which='both', direction='in', right=True,top=True)
    axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    axes[i].yaxis.set_minor_locator(AutoMinorLocator())
    axes[i].minorticks_on()
#axes[0].set_xlim(-1.0, 4.0)
axes[0].set_xlim(-0.5, 2.0)
axes[0].set_ylabel(r'$\langle N_{ee}\rangle/{\rm Tr}[N]$')
axes[1].set_ylabel(r'$|N_{ex}|/{\rm Tr}[N]$')

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("Nee_Nex_tr_2comp_fid.pdf", bbox_inches="tight")
