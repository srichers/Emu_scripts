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
Nee_str = ["N00_FFT", "N00_FFT(cm^-2)"]
Nxx_str = ["N11_FFT", "N11_FFT(cm^-2)"]
Nex_str = ["N01_FFT", "N01_FFT(cm^-2)"]
Nexavg_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]


def offdiagMag(f):
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


######################
# read averaged data #
######################
def plotdata(filename_FFT, filename_avg, t_in, ind_FFT, ind_avg, itmax=-1):
    if not os.path.exists(filename_FFT):
        print('No file')
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    t=np.array(fftData[t_str[ind_FFT]])
    k=np.array(fftData[k_str[ind_FFT]])
    #convert from 1/\lambda to k=2\pi/\lambda:
    k = 2.0*np.pi*k
    Nee=np.array(fftData[Nee_str[ind_FFT]])
    Nxx=np.array(fftData[Nxx_str[ind_FFT]])
    Nex=np.array(fftData[Nex_str[ind_FFT]])
    fftData.close()

    avgData = h5py.File(filename_avg,"r")
    t=np.array(avgData[t_str[ind_avg]])
    Nexavg=np.array(avgData[Nexavg_str[ind_avg]][:,0,1])
    avgData.close()

    # make time relative to tmax
    if itmax == -1:
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


fig, axes = plt.subplots(2,1, figsize=(6,10), sharex=True)
plt.subplots_adjust(hspace=0)

##############
# formatting #
##############
for ax in axes.flatten():
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    #ax.grid(which='both')
axes[1].set_xlabel(r"$k\,({\rm cm}^{-1})$")
axes[0].set_ylabel(r"$\mathcal{D}(k)\,\,{\rm [Fiducial]}$")
axes[1].set_ylabel(r"$\mathcal{D}(k)\,\,{\rm [90Degree]}$")
axes[0].set_xlim(0,8.0*2.0*np.pi)
axes[0].set_ylim(1.e-19,1.e-3)
axes[1].set_ylim(1.e-19,1.e-3)
#axes[0].text(6.5, 1.e-3, r"${\rm Fiducial}$")
#axes[1].text(6.5, 1.e-3, r"${\rm 90Degree}$")

#############
# plot data #
#############
tplot_list = [-0.1e-9, -0.1e-9]
basedirs = ["/global/cfs/projectdirs/m3761/FLASH/Emu/",
            "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/"]
savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/3tests/MPC/emu_comp/"
simlist_fid = ["Fiducial_3D_2F", "fid/MPC/d_pert/t5/xy_large/sim"]
simlist_90deg = ["90Degree_3D_2F", "90d/MPC/d_pert/t3/xy_large/sim"]

test_titles = [r'${\rm Fiducial}$', r'${\rm 90Degree}$']

itmax_inds = -np.ones([2,2], dtype=np.int8)
itmax_inds[1,1] = 118

def makeplot(ax, simlist, ind, ind_pow, ind_avg):
    tplot = tplot_list[ind]

    filename_emu_2f = basedirs[0]+simlist[0]+"/reduced_data_fft_power.h5"
    filename_emu_2f_avg = basedirs[0]+simlist[0]+"/reduced_data.h5"
    print('Emu 2f')
    k1,N1 = plotdata(filename_emu_2f,filename_emu_2f_avg,tplot,ind_pow[0],ind_avg[0])
    ax.semilogy(k1, N1, 'k--', label=r'${\rm {\tt Emu}}$')
    k_ind = np.argmax(N1)
    print('k_ind = ', k_ind)
    print('k(k_ind) = ', k1[k_ind])

    filename_bang   = basedirs[1]+simlist[1]+"/reduced_data_fft_power.h5"
    filename_bang_avg   = basedirs[1]+simlist[1]+"/reduced_data.h5"
    print('Flash sim')
    k3,N3 = plotdata(filename_bang,filename_bang_avg,tplot,ind_pow[1],ind_avg[1],itmax=itmax_inds[ind,1])
    ax.semilogy(k3, N3, 'r-', label=r'${\rm {\tt FLASH}}$')
    k_ind = np.argmax(N3)
    print('k_ind = ', k_ind)
    print('k(k_ind) = ', k3[k_ind])

    #ax.set_title(test_titles[ind])
    ax.set_xlim(0.0,35.0)


ind_pow = np.zeros([2,2], dtype=np.int8)
ind_avg = np.zeros([2,2], dtype=np.int8)


print('Fid')
makeplot(axes[0],simlist_fid, 0, ind_pow[0,:], ind_avg[0,:])
print('90d')
makeplot(axes[1],simlist_90deg, 1, ind_pow[1,:], ind_avg[1,:])

#\kmax from LSA:
axes[0].axvline(3.82, color='green', label=None)
axes[1].axvline(2.85, color='green', label=None)
    
axes[0].legend(loc='upper right', fontsize=28, frameon=False)
plt.savefig(savedir + "Nex_FFT_fid90d_MPC.pdf", bbox_inches="tight")
