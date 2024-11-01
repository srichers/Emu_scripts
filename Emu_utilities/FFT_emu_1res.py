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
def plotdata(filename_FFT, filename_avg, t_in):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    t=np.array(fftData["t(s)"])
    k=np.array(fftData["k(1|cm)"])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData["N00_FFT(cm^-2)"])
    Nxx=np.array(fftData["N11_FFT(cm^-2)"])
    Nex=np.array(fftData["N01_FFT(cm^-2)"])
    fftData.close()

    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData["t(s)"])
    Nexavg=np.array(avgData["N_avg_mag(1|ccm)"][:,0,1])
    avgData.close()

    # make time relative to tmax
    itmax = np.argmax(Nexavg)
    t = t-t[itmax]
    
    # get time closest to t
    dt = np.abs(t-t_in)
    it = np.argmin(dt)
    #it = int(itmax/2)
    trace = Nee[it,0]+Nxx[it,0]
    print(it,t[it],trace)
    return k, (Nex/trace)[it, :-1]

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
ax.set_xlabel(r"$k\,({\rm cm}^{-1})$")
ax.set_ylabel(r"$|\widetilde{N}_{ex}|/\mathrm{Tr}(N)$")
#one time only:
#ax.set_ylabel(r"$|\widetilde{N}_{ee}|/\mathrm{Tr}(N)$")
#axes[0].set_xlim(0,8)
#ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############
tplot = -1.0e-10
#tplot = -1.9591999999999988e-10
#tplot = -1.8479999999999982e-10
#tplot = -1.0467999999999989e-10
#tplot = -1.0467999999999989e-10
#tplot = -5.1559999999999975e-11
#tplot = -8.559999999999813e-12
#tplot = 0.0
#tplot = 8.54e-12
#tplot = 4.265e-11
#tplot = 1.e-10

filename_emu   = "plt_reduced_data_fft_power.h5"
filename_emu_avg   = "plt_reduced_data.h5"

k3,N3 = plotdata(filename_emu,filename_emu_avg,tplot)
ax.semilogy(k3, N3, 'k-')
ind3 = np.argmax(N3)
print('emu', ind3, k3[ind3], N3[ind3])

fig.text(0.5, 0.74, r'$t-t_{{\rm sat}}\sim{}\,{{\rm ns}}$'.format(1.e+9*tplot))
fig.text(0.5, 0.64, r'$k_{{\rm max}}\sim{:.2}\,{{\rm cm}}^{{-1}}$'.format(float(k3[ind3])))

plt.savefig("Nex_FFT_1res.pdf", bbox_inches="tight")

