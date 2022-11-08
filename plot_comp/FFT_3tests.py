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
    trace = Nee[it,0]+Nxx[it,0]
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


fig, axes = plt.subplots(1,3, figsize=(18,6), sharey=True)
plt.subplots_adjust(hspace=0,wspace=0)

##############
# formatting #
##############
for ax in axes.flatten():
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    #ax.grid(which='both')
    ax.set_xlabel(r"$k\,({\rm cm}^{-1})$")
axes[0].set_ylabel(r"$\widetilde{N}_{ex}/\mathrm{Tr}(N)$")
axes[0].set_xlim(0,8.0*2.0*np.pi)
axes[1].set_xlim(0.01,8.0*2.0*np.pi)
axes[2].set_xlim(0.01,2.0*2.0*np.pi)
axes[0].set_ylim(1.e-20,1.0)
axes[0].text(6.5, 1.e-3, r"${\rm Fiducial}$")
axes[1].text(6.5, 1.e-3, r"${\rm 90Degree}$")
axes[2].text(3.0, 1.e-3, r"${\rm TwoThirds}$")

#############
# plot data #
#############
tplot_list = [-0.1e-9, -0.1e-9, -0.35e-9]
basedirs = ["/global/project/projectdirs/m3761/Evan/",
            "/global/project/projectdirs/m3761/Evan/",
            "/global/project/projectdirs/m3761/FLASH/FFI_3D/"]
simlist_fid = ["Fiducial_3D_2F", "Fiducial_3D_3F", "fid"]
simlist_90deg = ["90Degree_3D_2F", "90Degree_3D_3F", "90d"]
simlist_23 = ["TwoThirds_3D_2F", "TwoThirds_3D_3F", "2_3"]

def makeplot(ax, simlist, ind):
    tplot = tplot_list[ind]

    filename_emu_2f = basedirs[0]+simlist[0]+"/reduced_data_fft_power.h5"
    filename_emu_2f_avg = basedirs[0]+simlist[0]+"/reduced_data.h5"
    k1,N1 = plotdata(filename_emu_2f,filename_emu_2f_avg,tplot)
    ax.semilogy(k1, N1, 'k-', label=r'${\rm Emu\,\,(2f)}$')

    filename_emu_3f = basedirs[1]+simlist[1]+"_reduced_data_fft_power.h5"
    filename_emu_3f_avg = basedirs[1]+simlist[1]+"_reduced_data.h5"
    k2,N2 = plotdata(filename_emu_3f,filename_emu_3f_avg,tplot)
    ax.semilogy(k2, N2, 'k--', label=r'${\rm Emu\,\,(3f)}$')

    filename_bang   = basedirs[2]+simlist[2]+"/sim1/reduced_data_fft_power_nov4_test_hdf5_chk.h5"
    filename_bang_avg   = basedirs[2]+simlist[2]+"/sim1/reduced_data_nov4_test_hdf5_chk.h5"
    k3,N3 = plotdata(filename_bang,filename_bang_avg,tplot)
    ax.semilogy(k3, N3, 'r-', label=r'${\rm FLASH\,\,(2f)}$')

    filename_bang   = basedirs[2]+simlist[2]+"/res_test1/reduced_data_fft_power_nov4_test_hdf5_chk.h5"
    filename_bang_avg   = basedirs[2]+simlist[2]+"/res_test1/reduced_data_nov4_test_hdf5_chk.h5"
    k3,N3 = plotdata(filename_bang,filename_bang_avg,tplot)
    ax.semilogy(k3, N3, 'r-', alpha=0.25, label=None)

    filename_bang   = basedirs[2]+simlist[2]+"/res_test2/reduced_data_fft_power_nov4_test_hdf5_chk.h5"
    filename_bang_avg   = basedirs[2]+simlist[2]+"/res_test2/reduced_data_nov4_test_hdf5_chk.h5"
    k3,N3 = plotdata(filename_bang,filename_bang_avg,tplot)
    ax.semilogy(k3, N3, 'r-', alpha=0.25, label=None)

makeplot(axes[0],simlist_fid, 0)
makeplot(axes[1],simlist_90deg, 1)
makeplot(axes[2],simlist_23, 2)

axes[0].axvline(3.75, color='green', label=None)
axes[1].axvline(2.84, color='green', label=None)
axes[2].axvline(1.50, color='green', label=None)
    
axes[0].legend(loc='upper right', frameon=False)
plt.savefig("N_ex_FFT_3tests.pdf", bbox_inches="tight")
