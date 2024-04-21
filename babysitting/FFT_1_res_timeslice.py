import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
def plotdata(filename_FFT, filename_avg):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    t=np.array(fftData["t"])
    k=np.array(fftData["k"])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData["N00_FFT"])
    Nxx=np.array(fftData["N11_FFT"])
    trace = Nee[0,0]+Nxx[0,0]
    Nex=np.array(fftData["N01_FFT"])
    fftData.close()

    avgData = h5py.File(filename_avg,"r")
    Nexavg=np.array(avgData["N_avg_mag"][:,0,1])
    # make time relative to tmax
    itmax = np.argmax(Nexavg)
    t = t-t[itmax]
    avgData.close()
    
    return t, k, (Nex/trace)[:,:-1]

################
# plot options #
################
mpl.rcParams['font.size'] = 16
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
ax = fig.add_subplot(111, projection='3d')

##############
# formatting #
##############
#ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.minorticks_on()
ax.set_xlabel(r"$k\,({\rm cm}^{-1})$")
ax.set_ylabel(r"$t\,(10^{-9}\,{\rm s})$")
ax.set_zlabel(r"$\widetilde{N}_{ex}/\mathrm{Tr}(N)$")
#axes[0].set_xlim(0,8)
tend = 1.e-09
#ax.set_ylim(-0.1,tend)
ax.set_zlim(-15.0, -2.0)

#############
# plot data #
#############

filename_bang   = "reduced_data_fft_power.h5"
filename_bang_avg   = "reduced_data.h5"
t3,k3,N3 = plotdata(filename_bang,filename_bang_avg)
t3 = 1.e+9*t3
#t = t3
k = k3
N = np.log10(N3)
plot_list = [1, 15, 30]
t = np.empty([len(k)])
for i in plot_list:
    t[:] = t3[i]
    ax.plot3D(k, t, N[i,:])


#fig.text(0.5, 0.8, r'$\delta m^2=7.53\times10^{-5}\,{\rm eV}^2$')
#fig.text(0.5, 0.72, r'$\theta=0.587$')
#fig.text(0.5, 0.64, r'$t-t_{{\rm max}}\sim{}\,{{\rm ns}}$'.format(1.e+9*tplot))
#ax.legend(loc='upper right', frameon=False)
plt.savefig("Nex_FFT_1res_timeslice.pdf", bbox_inches="tight")

print(np.max(N))
