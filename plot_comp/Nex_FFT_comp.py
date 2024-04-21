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
    
t_str = ["t", "t(s)"]
k_str = ["k", "k(1|cm)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]
fft_str_00 = ["N00_FFT", "N00_FFT(cm^-2)"]
fft_str_11 = ["N11_FFT", "N11_FFT(cm^-2)"]
fft_str_01 = ["N01_FFT", "N01_FFT(cm^-2)"]

def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename_FFT, filename_avg, t_in, ind):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    #t=np.array(fftData[t_str[ind]])
    k=np.array(fftData[k_str[ind]])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData[fft_str_00[ind]])
    Nxx=np.array(fftData[fft_str_11[ind]])
    Nex=np.array(fftData[fft_str_01[ind]])
    fftData.close()

    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData[t_str[ind]])
    Nexavg=np.array(avgData[N_str[ind]][:,0,1])
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
#axes[0].set_xlim(0,8)
ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############

#fid
tplot = -0.1e-9
basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_2F/",
            "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_3F/",
            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/"]
simlist = ["./", "./", "MPC/d_pert/sim/"]
tind = [0, 0, 0]
savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/comp_emu/"

#90d:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_2F/",
#            "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_3F/",
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/"]
#simlist = ["./", "./", "MPC/d_pert/sim1/"]

#2_3:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_2F/",
#            "/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_3F/",
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/"]
#simlist = ["./", "./", "MPC/d_pert/sim/"]
#tind = [1, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/comp_emu/"

#NSM1:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/",
#            "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/",
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/"]
#simlist = ["merger_2F/", "merger_3F/", "MPC/d_pert/sim2/"]

#NSM_3:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_3/",
#            None,
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/"]
#simlist = ["32dir/merger_2F/", None, "MPC/d_pert/sim/"]
#tind = [1, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/comp_emu/"



#NSM3:
#filename_emu_2f = basedirs[0]+simlist[0]+"plt_reduced_data_fft_power.h5"
#all others:
filename_emu_2f = basedirs[0]+simlist[0]+"reduced_data_fft_power.h5"
#comment out for NSM3:
filename_emu_3f = basedirs[1]+simlist[1]+"reduced_data_fft_power.h5"
filename_bang   = basedirs[2]+simlist[2]+"reduced_data_fft_power.h5"

#fid, 90d and 2_3:
filename_emu_2f_avg = basedirs[0]+simlist[0]+"reduced_data.h5"
#NSM1:
#filename_emu_2f_avg = basedirs[0]+simlist[0]+"reduced_data_old.h5"
#NSM3:
#filename_emu_2f_avg = basedirs[0]+simlist[0]+"plt_reduced_data.h5"

#comment out for NSM3:
filename_emu_3f_avg = basedirs[1]+simlist[1]+"reduced_data.h5"
filename_bang_avg   = basedirs[2]+simlist[2]+"reduced_data.h5"

k1,N1 = plotdata(filename_emu_2f,filename_emu_2f_avg,tplot,tind[0])
ax.semilogy(k1, N1, 'k-', label=r'${\rm {\tt Emu}\,\,(2f)}$')
#comment out for NSM3:
k2,N2 = plotdata(filename_emu_3f,filename_emu_3f_avg,tplot,tind[1])
ax.semilogy(k2, N2, 'k--', label=r'${\rm {\tt Emu}\,\,(3f)}$')
k3,N3 = plotdata(filename_bang,filename_bang_avg,tplot,tind[2])
ax.semilogy(k3, N3, 'r-', label=r'${\rm {\tt FLASH}\,\,(2f)}$')
#Vertical line from LSA for fastet growing mode
#ax.axvline(5.64, color='g', label=None)

ax.legend(loc='upper right', frameon=False)
plt.savefig(savedir + "Nex_FFT_emu_comp.pdf", bbox_inches="tight")

ind1 = np.argmax(N1)
print('emu_2f', ind1, k1[ind1], N1[ind1])
#comment out for NSM3:
ind2 = np.argmax(N2)
print('emu_3f', ind2, k2[ind2], N2[ind2])
ind3 = np.argmax(N3)
print('flash', ind3, k3[ind3], N3[ind3])
