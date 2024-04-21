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
    #t=np.array(avgData["t(s)"])*1e9
    #N=np.array(avgData["N_avg_mag(1|ccm)"])[:,a,b]
    #stop_ind = 30
    #For NSM_2:
    #stop_ind = 100
    #t = t[:stop_ind]
    #N = N[:stop_ind]
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
#filename = "reduced_data_0_cell_0_0_0.h5"
#filename = "reduced_data_nov4_test_hdf5_chk.h5"
#filename = "reduced_data_NSM_sim.h5"
#filename = "reduced_data_NSM_sim_hdf5_chk.h5"
#filename = "plt_reduced_data.h5"

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

#indices for flash fid:
#ind1 = 16
#ind2 = 6
#indices for flash 90d:
#ind1 = 20
#ind2 = 9
#indices for flash 2/3:
#ind1 = 16
#ind2 = 4
#indices for emu fid:
#ind1 = 31
#ind2 = 27
#indices for emu 90d:
#ind1 = 40
#ind2 = 20
#indices for emu 2/3:
#ind1 = 100
#ind2 = 50

#indices for flash NSM_1:
#ind1 = 8
#ind2 = 3
#indices for flash NSM_2:
#ind1 = 46
#ind2 = 41
#indices for flash NSM_3:
#ind1 = 51
#ind2 = 46
#indices for emu NSM_1:
#ind1 = 15
#ind2 = 11
#indices for emu NSM_2:
#ind1 = 61
#ind2 = 21
#indices for emu NSM_3:
#ind1 = 21
#ind2 = 11

#indices for FLASH/FFI_1D/Beam/rand/MPC/
#sim1
#ind1 = 5
#ind2 = 10
#sim2
#ind1 = 3
#ind2 = 7
#sim3 & sim4
#ind1 = 2
#ind2 = 4

#indices for FLASH/FFI_1D/fid/MPC/
#sim1
#ind1 = 10
#ind2 = 20
#sim2
#ind1 = 6
#ind2 = 10
#sim3 & sim4
#ind1 = 4
#ind2 = 6

#indices for FLASH/FFI_3D/fid/MPC/
#sim1
#ind1 = 11
#ind2 = 20

##determine indices
##find maximum of N:
#indmax = np.argmax(N)
##ind1 should be close to ~10 larger than initial input:
##N_lower = 10.0*N[0]
##ind1 = np.argmin(np.abs(np.log(N[0:indmax+1]/N_lower)))
#N_lower = 1.e-10
#ind1 = np.argmin(np.abs(np.log(N[1:indmax+1]/N_lower)))
##ind2 should be close to ~10 smaller than max value:
#N_upper = 0.1*N[indmax]
##ind2 = np.argmin(np.abs(np.log(N[0:indmax+1]/N_upper)))
#ind2 = np.argmin(np.abs(np.log(N[1:indmax+1]/N_upper)))

indmax = np.argmax(N)
N_lower = 1.e-5
N_upper = 1.e-1
ind1 = np.argmin(np.abs(np.log(N[1:indmax+1]/N_lower)))
ind2 = np.argmin(np.abs(np.log(N[1:indmax+1]/N_upper)))
#indices used for fid/sim
ind1 = 33
ind2 = 42
#indices used for 2_3/res_a
#ind1 = 30
#ind2 = 36
#indices used for NSM_1/t3/res_b
#ind1 = 13
#ind2 = 19
scale_fact = 10.0

#indices needed for SDA test (periodicity causes false maxima):
#ind1 = 3
#ind2 = 10

ot_est = (np.log(N[ind1]) - np.log(N[ind2]))/1.e-9/(t[ind1] - t[ind2])
t_line = [t[ind2], t[ind1]]
N_line = [scale_fact*N[ind2], scale_fact*N[ind1]]
ax.semilogy(t_line, N_line, color='orange')
ax.set_title(r"$\tilde{{\omega}}={:.2E}$".format(ot_est))
plt.savefig("avgfemu_est.pdf", bbox_inches="tight")
#plt.savefig("../../temp/avgfemu_est.pdf", bbox_inches="tight")
print('Nex/Tr[0]: ', N[0])
print('Nex/Tr max: ', N[indmax])
print('Indices used: ', ind1, ind2)
print('Nex/Tr values: ', N[ind1], N[ind2])
print('Estimated growth rate = ', 1.e-10*ot_est, 'e10 1/s')
