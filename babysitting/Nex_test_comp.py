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
    if nu_type == 1:
        N=np.array(avgData["Nbar_avg_mag"])[:,a,b]
    else:
        N=np.array(avgData["N_avg_mag"])[:,a,b]
    avgData.close()
    return t, N
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
#filename = "reduced0D_selection.h5"

##############
# formatting #
##############
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
ax.xaxis.set_minor_locator(AutoMinorLocator())
if nu_type == 1:
    ax.set_ylabel(r"$\langle |\overline{N}_{ex}|\rangle /\mathrm{Tr}(\overline{N})$")
else:
    ax.set_ylabel(r"$\langle |N_{ex}|\rangle /\mathrm{Tr}(N)$")
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.grid(which='both')

# same for f_e\mu
t,N = plotdata(filename,0,1)

tmax = t[np.argmax(N)]

H_nu = -3.019737074078213e-11 #MeV
H_M = 5.5790200754556285e-09 #MeV
hbar = 4.135667696e-21 #Mev*s
omega_v = np.sqrt(-H_nu*H_M)/hbar
print("V geometric average", omega_v/1.e9, "e9 1/s")


#amp = 1.e-2
#offset = 5.e-2
#t_test = np.linspace(1.0,3.0, 1000) #ns
#omega_test = 40.e+9/1.e9 #divide because t_test is in ns (for xlim=[1.0,3.0])
#ax.semilogy(t, N)
#ax.semilogy(t_test, -amp*np.cos(omega_test*(t_test-tmax)) + offset, color='orange')
#ax.set_xlim(1.0,3.0)

#amp = 3.e-3
#offset = 7.e-3
#t_test = np.linspace(0.0,1.0, 1000) #ns
#omega_test = 65.e+9/1.e9 #divide because t_test is in ns (for xlim=[0.0,1.0])
#ax.semilogy(t, N)
##ax.semilogy(t_test, amp*np.cos(omega_test*(t_test-0.929)) + offset, color='orange')
#ax.semilogy(t_test, amp*np.cos(omega_test*(t_test)) + offset, color='orange')
#ax.set_xlim(0.0,1.0)

#amp = 5.e-8
#offset = 1.e-7
#t_test = np.linspace(0.0,2.0, 1000) #ns
#omega_test = 62.e+9/1.e9 #divide because t_test is in ns (for xlim=[0.0,1.0])
#tee, Nee = plotdata(filename,0,0)
#txx, Nxx = plotdata(filename,1,1)
#trace_N = Nee[0] + Nxx[0]
#ax.semilogy(t, N/trace_N)
#ax.semilogy(t_test, -amp*np.cos(omega_test*(t_test)) + offset, color='orange')
##ax.semilogy(t_test, amp*np.cos(omega_test*(t_test-0.929)) + offset, color='orange')
#ax.set_xlim(0.0,2.0)
#ax.set_ylim(1.e-9,1.e-6)


#NSM2.5/d_pert/t4/res_a
amp = 3.e-8
offset = 1.e-7
t_test = np.linspace(0.0,1.0, 1000) #ns
omega_test = 65.e+9/1.e9 #divide because t_test is in ns (for xlim=[0.0,1.0])
ax.semilogy(t, N)
#ax.semilogy(t_test, amp*np.cos(omega_test*(t_test-0.929)) + offset, color='orange')
ax.semilogy(t_test, amp*np.cos(omega_test*(t_test)) + offset, color='orange')
ax.set_xlim(0.0,1.0)

print("Test", omega_test, "e9 1/s")


ax.set_title(r"$\omega_{{\rm test}}={:.2E}\,{{\rm s}}^{{-1}}$".format(1.e9*omega_test))

if nu_type == 1:
    nu_type_str = 'bnu'
else:
    nu_type_str = 'nu'

namestr = "Nex_test_comp_" + nu_type_str + ".pdf"
plt.savefig(namestr, bbox_inches="tight")
