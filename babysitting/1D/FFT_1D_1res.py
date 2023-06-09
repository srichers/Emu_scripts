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
    t=np.array(fftData["t"])
    k=np.array(fftData["k"])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData["N00_FFT"])
    Nxx=np.array(fftData["N11_FFT"])
    Nex=np.array(fftData["N01_FFT"])
    fftData.close()

    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData["t"])
    Nexavg=np.array(avgData["N_avg_mag"][:,0,1])
    avgData.close()

    # make time relative to tmax
    itmax = np.argmax(Nexavg)
    #t = t-t[itmax]
    
    # get time closest to t
    #dt = np.abs(t-t_in)
    #it = np.argmin(dt)
    it = itmax
    trace = Nee[it,0]+Nex[it,0]
    #print(it,t[it], Nexavg[itmax])
    print(it,t[it], Nexavg[it])
    return k, (Nex/trace)[it, :-1], t[it], it

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
ax.set_ylabel(r"$\widetilde{N}_{ex}/\mathrm{Tr}(N)$")
ax.set_xlim(-10,210)
ax.set_ylim(1.e-16,1.0)

box_length = 8.0
n_blks = 64
cells_per_blk = 16

fig.text(0.5, 0.77, r'$L={:.3f}\,{{\rm cm}}$'.format(box_length))
fig.text(0.5, 0.67, r'$N_{{B}}={}$'.format(n_blks))
fig.text(0.5, 0.57, r'$N_{{gp}}={}$'.format(n_blks*cells_per_blk))

#############
# plot data #
#############
#tplot = -0.1e-9
tplot = 0.0

#filename_bang   = "reduced_data_fft_power_NSM_sim.h5"
#filename_bang_avg   = "reduced_data_NSM_sim.h5"
filename_bang   = "reduced_data_fft_power.h5"
filename_bang_avg   = "reduced_data.h5"
#filename_bang   = "reduced_data_fft_power_NSM_res2.h5"
#filename_bang_avg   = "reduced_data_NSM_res2.h5"
k3,N3, tdata, it = plotdata(filename_bang,filename_bang_avg,tplot)
ax.semilogy(k3, N3, 'r-', label=r'${\rm {\tt FLASH}\,\,(2f)}$')
#Vertical line from LSA for fastet growing mode
ax.set_title(r'$t={:.2}\times 10^{{-11}}\,{{\rm s}}$'.format(1.e11*tdata))
#ax.axvline(5.64, color='g', label=None)

#ax.legend(loc='upper right', frameon=False)
#plt.savefig("Nex_FFT_1res_1D_0.pdf", bbox_inches="tight")
namestr = "Nex_FFT_1res_1D_{}.pdf".format(it)
plt.savefig(namestr, bbox_inches="tight")

ind3 = np.argmax(N3)
print('flash', ind3, k3[ind3], N3[ind3])
