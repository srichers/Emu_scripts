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
def plotdata(filename_FFT, filename_avg, t_in, ind_fft=0, ind_avg=0, itmax_in=-1):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    #t=np.array(fftData[t_str[ind_fft]])
    k=np.array(fftData[k_str[ind_fft]])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData[fft_str_00[ind_fft]])
    Nxx=np.array(fftData[fft_str_11[ind_fft]])
    Nex=np.array(fftData[fft_str_01[ind_fft]])
    fftData.close()

    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData[t_str[ind_avg]])
    Nexavg=np.array(avgData[N_str[ind_avg]][:,0,1])
    avgData.close()

    # make time relative to tmax
    #special code for NSM_2:
    if itmax_in == -1:
        itmax = np.argmax(Nexavg)
    else:
        itmax = itmax_in
    t = t-t[itmax]
    
    # get time closest to t
    dt = np.abs(t-t_in)
    it = np.argmin(dt)
    trace = Nee[0,0] + Nxx[0,0]
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


#############
# plot data #
#############

fftind = [0, 0, 0]
avgind = [0, 0, 0]
itmax_inds = [-1, -1, -1]

#fid
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_2F/",
#            "/global/cfs/projectdirs/m3761/FLASH/Emu/Fiducial_3D_3F/",
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/"]
#simlist = ["./", "./", "MPC/d_pert/t4/xy_large/sim/"]
#tind = [0, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/t4/xy_large/comp_emu/"

##90d:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_2F/",
#            "/global/cfs/projectdirs/m3761/FLASH/Emu/90Degree_3D_3F/",
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/"]
#simlist = ["./", "./", "MPC/d_pert/t2/xy_large/sim/"]
#tind = [0, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/t2/xy_large/comp_emu/"

#2_3:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_2F/",
#            "/global/cfs/projectdirs/m3761/FLASH/Emu/TwoThirds_3D_3F/",
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/"]
#simlist = ["./", "./", "MPC/d_pert/t2/xy_large/sim/"]
#avgind = [1, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/t2/xy_large/comp_emu/"

#NSM1:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/",
#            None,
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/"]
#simlist = ["merger_2F/", "merger_3F/", "MPC/d_pert/t3/sim/"]
#tind = [0, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/comp_emu/"
#est_kmax = 5.68 #cm^{-1}
#ax_title = r"${\rm NSM}1$"
#ylim = (1.e-18, 1.e-6)

#NSM_3:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_3/",
#            None,
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/"]
#simlist = ["32dir/merger_2F/", None, "MPC/d_pert/sim/"]
#tind = [1, 0, 0]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/comp_emu/"


#NSM_3/t3 xy_large:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_3/",
#            None,
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/"]
#simlist = ["32dir/merger_2F/", None, "MPC/d_pert/t3/xy_large/sim/long_t/"]
#tind = [1, 0, 0]
#fftind[0] = 1
#avgind[0] = 1
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/xy_large/comp_emu/"
#est_kmax = 6.6 #cm^{-1}
#ax_title = r"${\rm NSM}3$"
#xlim = (-1.0, 40.0)

#NSM_2/clos3/t4 xy_large:
tplot = -0.1e-9
basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_2/",
            None,
            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2/"]
simlist = ["32dir/merger_2F/", None, "MPC/clos3/d_pert/t4/xy_large/sim/"]
tind = [1, 0, 0]
fftind[0] = 1
avgind[0] = 1
itmax_inds[2] = 235
savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2/MPC/clos3/d_pert/t4/xy_large/comp_emu/"
emu_test = "NSM_2"
emu_type = "sim"
xlim = (-2.0, 50.0)
est_kmax = 3.7 #cm^{-1}
ax_title = r"${\rm NSM}2$"

#NSM_2.5/t2/xy_large, and Emu/NSM_2.5/1res/correct_Nxx/NSM_2.5_matchEvan*/plt**:
#tplot = -0.1e-9
#basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_2.5/1res/correct_Nxx/NSM2.5_matchevan_correctNxx_long_diagonalpert/",
#            None,
#            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/"]
#simlist = ["./", None, "./"]
#tind = [1, 0, 0]
#fftind[0] = 1
#avgind[0] = 1
#itmax_inds = [-1, -1, -1]
#savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/comp_emu/plt/"
#emu_test = "NSM_2.5"

#NSM3 and NSM2 NSM2.5:
filename_emu_2f = basedirs[0]+simlist[0]+"plt_reduced_data_fft_power.h5"
#all others:
#filename_emu_2f = basedirs[0]+simlist[0]+"reduced_data_fft_power.h5"
#comment out for NSM3:
#filename_emu_3f = basedirs[1]+simlist[1]+"reduced_data_fft_power.h5"
filename_bang   = basedirs[2]+simlist[2]+"reduced_data_fft_power.h5"

#fid, 90d and 2_3:
#filename_emu_2f_avg = basedirs[0]+simlist[0]+"reduced_data.h5"
#NSM1:
#filename_emu_2f_avg = basedirs[0]+simlist[0]+"reduced_data_old.h5"
#NSM3 and NSM2:
filename_emu_2f_avg = basedirs[0]+simlist[0]+"plt_reduced_data.h5"

#comment out for NSM3:
#filename_emu_3f_avg = basedirs[1]+simlist[1]+"reduced_data.h5"
filename_bang_avg   = basedirs[2]+simlist[2]+"reduced_data.h5"


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
#ax.set_ylabel(r"$|\widetilde{N}_{ex}|^2/\langle N_{ee}^2(t=0) + N_{xx}^2(t=0)\rangle$")
ax.set_ylabel(r"$\mathcal{D}(k)$")
if "xlim" in locals():
    ax.set_xlim(xlim[0], xlim[1])
if "ylim" in locals():
    ax.set_ylim(ylim[0], ylim[1])


k1,N1 = plotdata(filename_emu_2f,filename_emu_2f_avg,tplot,fftind[0],avgind[0], itmax_inds[0])
ax.semilogy(k1, N1, 'k--', label=r'${\rm {\tt Emu}}$')
#comment out for NSM3 and NSM2.5:
#k2,N2 = plotdata(filename_emu_3f,filename_emu_3f_avg,tplot,fftind[1],avgind[1])
#ax.semilogy(k2, N2, 'k--', label=r'${\rm {\tt Emu}\,\,(3f)}$')
k3,N3 = plotdata(filename_bang,filename_bang_avg,tplot, fftind[2], avgind[2], itmax_inds[2])
ax.semilogy(k3, N3, 'r-', label=r'${\rm {\tt FLASH}}$')

#Vertical line from LSA for fastet growing mode
if 'est_kmax' in locals():
    ax.axvline(est_kmax, color='g', label=None)

if 'ax_title' in locals():
    ax.set_title(ax_title)

ax.legend(loc='upper right', frameon=False)
plt.savefig(savedir + "Nex_FFT_emu_comp.pdf", bbox_inches="tight")

ind1 = np.argmax(N1)
print('emu_2f', ind1, k1[ind1], N1[ind1])
#comment out for NSM3 and NSM2.5:
#ind2 = np.argmax(N2)
#print('emu_3f', ind2, k2[ind2], N2[ind2])
ind3 = np.argmax(N3)
print('flash', ind3, k3[ind3], N3[ind3])
