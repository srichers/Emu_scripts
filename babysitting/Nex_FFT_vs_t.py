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
def plotdata(filename_FFT):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    t=np.array(fftData["t"])
    k=np.array(fftData["k"])
    #convert to ns
    t = 1.e+09*t
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData["N00_FFT"])
    Nxx=np.array(fftData["N11_FFT"])
    Nex=np.array(fftData["N01_FFT"])
    fftData.close()

    trace = Nee[0,np.argmin(np.abs(k))]+Nxx[0,np.argmin(np.abs(k))]
    print(trace)
    return t, k, (Nex/trace)

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


fig = plt.figure()
ax = fig.gca()

##############
# formatting #
##############
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.set_xlabel(r"$t\,(10^{-9}\,{\rm s}$")
ax.set_ylabel(r"$|\widetilde{N}_{ex}|^2/(N_{ee}^2 + N_{xx}^2)$")
#one time only:
#ax.set_ylabel(r"$|\widetilde{N}_{ee}|/\mathrm{Tr}(N)$")
#axes[0].set_xlim(0,8)
#ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############
tplot = -0.5e-10

filename_bang   = "reduced_data_fft_power.h5"

#2/3 t3/xy_large/sim:
ind1 = 0
ind2 = 6

t, k3, N3 = plotdata(filename_bang)
ax.semilogy(t, N3[:,ind1], 'r-', label=r'$k={}$'.format(k3[ind1]))
ax.semilogy(t, N3[:,ind2], 'b--', label=r'$k={:.2}$'.format(k3[ind2]))
#ax.set_xlim(-10.0,10.0)
#ax.set_ylim(1.e-7,5.e-3)

print("ind1 = ", ind1, " k[ind1] = ", k3[ind1])
print("ind2 = ", ind2, " k[ind2] = ", k3[ind2])

#fig.text(0.5, 0.8, r'$\delta m^2=7.53\times10^{-5}\,{\rm eV}^2$')
#fig.text(0.5, 0.72, r'$\theta=0.587$')
#fig.text(0.15, 0.74, r'$t-t_{{\rm sat}}\sim{}\,{{\rm ns}}$'.format(1.e+9*tplot))
#fig.text(0.5, 0.74, r'$t\sim0\,{{\rm ns}}$')
#fig.text(0.15, 0.64, r'$k_{{z,{{\rm max}}}}\sim{:.2}\,{{\rm cm}}^{{-1}}$'.format(float(k3[ind3])))
ax.legend(loc='lower right', frameon=False)

plt.savefig("Nex_FFT_vs_t.pdf", bbox_inches="tight")

