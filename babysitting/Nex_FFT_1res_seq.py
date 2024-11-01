import os
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
def plotdata(filename_FFT, filename_avg, t_index):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]

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


##############
# formatting #
##############
#one time only:
#axes[0].set_xlim(0,8)
#ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############
tplot = -0.5e-10

#basename = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/"
#filename_FFT   = basename + "reduced_data_fftz_power.h5"
#filename_bang_avg   = "reduced_data.h5"
basename = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/res_1024/sim0.2/"
filename_FFT   = basename + "reduced_data_fft_power.h5"
#basename = "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_2.5/1res/correct_Nxx/NSM2.5_matchevan_correctNxx_long_diagonalpert/"
#filename_FFT   = basename + "plt_reduced_data_fft_power.h5"

fftData = h5py.File(filename_FFT,"r")
t=np.array(fftData["t"])
k=np.array(fftData["k"])
#t=np.array(fftData["t(s)"])
#k=np.array(fftData["k(1|cm)"])
#convert from 1/\lambda to k=2\pi/\lambda:
k = 2.0*np.pi*k

Nee=np.array(fftData["N00_FFT"])
Nxx=np.array(fftData["N11_FFT"])
Nex=np.array(fftData["N01_FFT"])
#Nee=np.array(fftData["N00_FFT(cm^-2)"])
#Nxx=np.array(fftData["N11_FFT(cm^-2)"])
#Nex=np.array(fftData["N01_FFT(cm^-2)"])

seq_int = 386
kmax = np.empty([seq_int-1])
N3k0 = np.empty([seq_int-1])
t_for_kmax = np.empty([seq_int-1])

#for i in range(1,seq_int):
for i in range(0,seq_int):

    fig = plt.figure(i)
    ax = fig.gca()
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.set_xlabel(r"$k\,({\rm cm}^{-1})$")
    ax.set_ylabel(r"$\mathcal{D}(k)$")

    trace = Nee[i,np.argmin(np.abs(k))]+Nxx[i,np.argmin(np.abs(k))]
    N3 = (Nex/trace)[i, :-1]
    ind3 = np.argmax(N3)
    ax.semilogy(k, N3, 'r-')
    print(i, 't=', 1.e+9*t[i], ' ns, index for fgm:', ind3, ', kmax = ', k[ind3], ' 1/cm, (N_ex/Tr)**2 value:', N3[ind3])
    fig.text(0.56, 0.75, r'$t = {:.3}\,{{\rm ns}}$'.format(1.e+9*t[i]))
    fig.text(0.56, 0.65, r'$k_{{{{\rm max}}}}\sim{:.2}\,{{\rm cm}}^{{-1}}$'.format(float(k[ind3])))
    
    plt.savefig(basename + "seq/Nex_FFT_1res_tind_{0:04d}.pdf".format(i), bbox_inches="tight")

    fig.clf()

    t_for_kmax[i-1] = t[i]
    kmax[i-1] = k[ind3]
    N3k0[i-1] = N3[np.argmin(k)]

fftData.close()

fig = plt.figure(seq_int+1)
ax = fig.gca()
ax.plot(1.e+9*t_for_kmax, kmax, 'b-')
ax.set_xlabel(r"$t\,({\rm ns})$")
ax.set_ylabel(r"$k_{z,{\rm max}}\,({\rm cm})^{-1}$")
plt.savefig(basename + "seq/Nex_FFT_1res_kmax_vs_t.pdf", bbox_inches="tight")
fig.clf()

fig = plt.figure(seq_int+2)
ax = fig.gca()
ax.semilogy(1.e+9*t_for_kmax, N3k0, 'r-')
ax.set_xlabel(r"$t\,({\rm ns})$")
ax.set_ylabel(r"$|\widetilde{N}_{ex}(k_z=0)|^2/(N_{ee}^2 + N_{xx}^2)$")
plt.savefig(basename + "seq/Nex_FFT_1res_N3k0_vs_t.pdf", bbox_inches="tight")
fig.clf()
