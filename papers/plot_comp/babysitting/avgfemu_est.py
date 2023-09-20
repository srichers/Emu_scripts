# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)


base=["N","Fx","Fy","Fz"]
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
    stop_ind = 30
    t = t[:stop_ind]
    N = N[:stop_ind]
    avgData.close()
    return t, N

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
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

filename = "reduced_data.h5"
#filename = "reduced_data_nov4_test_hdf5_chk.h5"
#filename = "reduced_data_NSM_sim.h5"
#filename = "reduced_data_NSM_sim_hdf5_chk.h5"

##############
# formatting #
##############
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.set_ylabel(r"$\langle N_{ex}\rangle /\mathrm{Tr}(N)$")
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

# same for f_e\mu
t,N = plotdata(filename,0,1)
ax.semilogy(t, N)
#original indices used:
#ind1 = 5
#ind2 = 1
#indices for flash NSM:
#ind1 = 8
#ind2 = 3
#indices for emu NSM:
ind1 = 15
ind2 = 11
ot_est = (np.log(N[ind1]) - np.log(N[ind2]))/1.e-9/(t[ind1] - t[ind2])
t_line = [t[ind2], t[ind1]]
N_line = [10.0*N[ind2], 10.0*N[ind1]]
ax.semilogy(t_line, N_line, color='orange')
ax.set_title(r"$\tilde{{\omega}}={:.2E}$".format(ot_est))
plt.savefig("../../avgfemu_est.pdf", bbox_inches="tight")
#plt.savefig("avgfemu_est.pdf", bbox_inches="tight")
