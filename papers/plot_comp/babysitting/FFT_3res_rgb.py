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
    t = t-t[itmax]
    
    # get time closest to t
    dt = np.abs(t-t_in)
    it = np.argmin(dt)
    trace = Nee[it,0]+Nex[it,0]
    print(it,t[it])
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
ax.set_ylabel(r"$\widetilde{N}_{ex}/\mathrm{Tr}(N)$")
#axes[0].set_xlim(0,8)
#ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############
tplot = -0.1e-9

basedir = '/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/t6/'
plotdir = '/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/t6/sim_res_comp/'
#t1
#box_length = 5.803047682077095 #cm
#n_grid = 128
#t2
#box_length = 5.803047682077095 #cm
#n_grid = 256
#t3
#box_length = 2.9015238410385473 #cm
#n_grid = 128
#t4
#box_length = 11.60609536415419 #cm
#n_grid = 256
#t5
#box_length = 11.60609536415419 #cm
#n_grid = 512
#t6
box_length = 5.803047682077095 #cm
n_grid = 512

simres = ['sim/', 'res_a/', 'res_b/']

filename_bang   = basedir + simres[0] + "reduced_data_fft_power.h5"
filename_bang_avg   = basedir + simres[0] + "reduced_data.h5"
k0,N0 = plotdata(filename_bang,filename_bang_avg,tplot)
ax.semilogy(k0, N0, 'r-', label = r'$N_{{gp}}/L={}/{:.3f}\,{{\rm cm}}$'.format(n_grid,box_length))

filename_bang   = basedir + simres[1] + "reduced_data_fft_power.h5"
filename_bang_avg   = basedir + simres[1] + "reduced_data.h5"
k1,N1 = plotdata(filename_bang,filename_bang_avg,tplot)
ax.semilogy(k1, N1, 'g-', label = r'$N_{{gp}}/L={}/{:.3f}\,{{\rm cm}}$'.format(n_grid/2,box_length/2.0))

filename_bang   = basedir + simres[2] + "reduced_data_fft_power.h5"
filename_bang_avg   = basedir + simres[2] + "reduced_data.h5"
k2,N2 = plotdata(filename_bang,filename_bang_avg,tplot)
ax.semilogy(k2, N2, 'b-', label = r'$N_{{gp}}/L={}/{:.3f}\,{{\rm cm}}$'.format(n_grid/2,box_length))

#Vertical line from LSA for fastet growing mode
#ax.axvline(5.64, color='g', label=None)

ax.legend(loc='upper right', fontsize=14, frameon=False)
plt.savefig(plotdir + "Nex_FFT_3res_rgb.pdf", bbox_inches="tight")
