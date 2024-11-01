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
    t=np.array(avgData["t(s)"])*1e9
    N=np.array(avgData["N_avg_mag(1|ccm)"])[:,a,b]
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

##############
# formatting #
##############
#ax.axhline(1./3., color="green")
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

#############
# plot data #
#############
filename = "plt_reduced_data.h5"
t,N = plotdata(filename,0,0)
N0 = N[0]
#ax.plot(t, N/N[0])
#ax.set_ylabel(r"$\langle N_{ee}\rangle /N(0)$")
#plt.savefig("avgfee.pdf", bbox_inches="tight")

t,N = plotdata(filename,0,1)
N = N/N0
#calculate Im(\Omega)_max
indmax = np.argmax(N)
#NSM_2.5/1res/correct_Nxx/NSM2.5_matchevan_correctNxx_long_diagonalpert/
N_lower = 5.e-7
N_upper = 1.e-2
scale_fact = 10.0
ind1 = np.argmin(np.abs(np.log(N[1:indmax+1]/N_lower)))
ind2 = np.argmin(np.abs(np.log(N[1:indmax+1]/N_upper)))
ot_est = (np.log(N[ind1]) - np.log(N[ind2]))/1.e-9/(t[ind1] - t[ind2])
t_line = [t[ind2], t[ind1]]
N_line = [scale_fact*N[ind2], scale_fact*N[ind1]]
ax.semilogy(t, N)
ax.semilogy(t_line, N_line, color='orange')
ax.set_title(r"$\tilde{{\omega}}={:.2E}$".format(ot_est))

ax.set_ylabel(r"$\langle N_{ex}\rangle /N(0)$")
plt.savefig("avgfemu.pdf", bbox_inches="tight")

print('Nex/Nee[0]: ', N[0])
print('Nex/Nee max: ', N[indmax])
print('Indices used: ', ind1, ind2)
print('Nex/Nee values: ', N[ind1], N[ind2])
print('Estimated growth rate = ', 1.e-10*ot_est, 'e10 1/s')
