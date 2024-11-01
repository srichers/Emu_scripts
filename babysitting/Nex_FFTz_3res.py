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
def plotdata(filename_FFT, filename_avg, t_in, itmax=-1):
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
    if itmax == -1:
        itmax = np.argmax(Nexavg)
    t = t-t[itmax]
    
    # get time closest to t
    dt = np.abs(t-t_in)
    it = np.argmin(dt)
    trace = Nee[it,np.argmin(np.abs(k))]+Nxx[it,np.argmin(np.abs(k))]
    print(it,t[it])
    return k, (Nex/trace)[it, :]

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
ax.set_xlabel(r"$k_z\,({\rm cm}^{-1})$")
ax.set_ylabel(r"$\mathcal{D}(k_z)$")
#axes[0].set_xlim(0,8)
#ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############

simres = ['sim/', 'res_a/', 'res_b/']
savedirname = "comp_res/Nex_FFTz_3res.pdf"
sim_name = "FFI_3D"
itmax_inds = np.array([-1, -1, -1])

#fid:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/fid/MPC/d_pert/t4/xy_large/"
#labels = [r'$N_{gp}=128^3;\,L=8.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=4.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=8.0\,{\rm cm}$']
#savedirname = "comp_res/Nex_FFT_3res.pdf"
#est_kmax = 3.82 #cm^{-1}

##90d:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/90d/MPC/d_pert/t2/xy_large/"
#labels = [r'$N_{gp}=64^3;\,L=8.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=4.0\,{\rm cm}$', \
#    r'$N_{gp}=128^3;\,L=8.0\,{\rm cm}$']
#savedirname = "comp_res/Nex_FFT_3res.pdf"
#est_kmax = 2.85 #cm^{-1}

##2_3:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/2_3/MPC/d_pert/t2/xy_large/"
#labels = [r'$N_{gp}=128^3;\,L=32.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=16.0\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=32.0\,{\rm cm}$']
#savedirname = "comp_res/Nex_FFT_3res.pdf"
##from LSA:
#est_kmax = 1.49 #cm^{-1}

#NSM_1/t2:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/"
#simres = ['sim2/', 'res_a2/', 'res_b2/']
#labels = [r'$N_{gp}=64^3;\,L=7.87\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=3.93\,{\rm cm}$'), \
#    r'$N_{gp}=128^3;\,L=7.87\,{\rm cm}$']

#NSM_1/t3:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/"
#simres = ['sim/', 'res_a/', 'res_b/']
#labels = [r'$N_{gp}=128^3;\,L=7.87\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=3.93\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=7.87\,{\rm cm}$']
#savedirname = "comp_res/Nex_FFT_3res.pdf"
#est_kmax = 5.68 #cm^{-1}

#NSM_3:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/"
#simres = ['sim/', 'res_a/', 'res_b/']
#labels = [r'$N_{gp}=128^3;\,L=5.80\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=2.90\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=5.80\,{\rm cm}$']

#NSM_3/t3:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/"
#simres = ['sim_xmax_large/', 'res_a_xmax_large/', 'res_b_xmax_large/']
#labels = [r'$N_{gp}=16^2\times512;\,L=5.80\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times256;\,L=2.90\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times256;\,L=5.80\,{\rm cm}$']
#savedirname = "comp_res/xy_large/Nex_FFT_3res.pdf"
#est_kmax = 6.6 #cm^{-1}

#NSM_2.5/t1:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t1/"
#labels = [r'$N_{gp}=128^3;\,L=24.5\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=12.3\,{\rm cm}$', \
#    r'$N_{gp}=64^3;\,L=24.5\,{\rm cm}$']
#est_kmax = 2.05 #cm^{-1}

#NSM_2.5/t2/xy_[large,small]/:
sim_name = "NSM_2.5"
tplot = -0.1e-9
basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/"
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_small/"
labels = [r'$N_{gp}=16^2\times256$', \
    r'$N_{gp}=16^2\times128$', \
    r'$N_{gp}=16^2\times128$']
est_kmax = 2.05 #cm^{-1}
#x_limits = (-10.0, 10.0)

##NSM_2.5/od_pert/t2/xy_large/:
#tplot = -0.1e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/od_pert/t2/xy_large/"
#labels = [r'$N_{gp}=16^2\times256$', \
#    r'$N_{gp}=16^2\times128$', \
#    r'$N_{gp}=16^2\times128$']
#est_kmax = 2.05 #cm^{-1}

##NSM_2/MPC/clos3/d_pert/t[1-3]/xy_large/:
#sim_name = "NSM_2"
#tplot = -0.2e-9
#basedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2/MPC/clos3/d_pert/t3/xy_large/"
##t1:
##labels = [r'$N_{gp}=16^2\times256;\,L=8.27\,{\rm cm}$', \
##    r'$N_{gp}=16^2\times128;\,L=4.13\,{\rm cm}$', \
##    r'$N_{gp}=16^2\times128;\,L=8.27\,{\rm cm}$']
##t2:
##labels = [r'$N_{gp}=16^2\times512;\,L=8.27\,{\rm cm}$', \
##    r'$N_{gp}=16^2\times256;\,L=4.13\,{\rm cm}$', \
##    r'$N_{gp}=16^2\times256;\,L=8.27\,{\rm cm}$']
##itmax_inds[0] = 245
##t3:
#labels = [r'$N_{gp}=16^2\times512;\,L=16.53\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times256;\,L=8.27\,{\rm cm}$', \
#    r'$N_{gp}=16^2\times256;\,L=16.53\,{\rm cm}$']
#itmax_inds[0] = 126
#est_kmax = 3.7 #cm^{-1}
#x_limits = (-10.0, 10.0)

fft_name = "reduced_data_fftz_power.h5"
avg_name = "reduced_data.h5"

filename_bang   = basedir + simres[2] + fft_name
filename_bang_avg   = basedir + simres[2] + avg_name
k2,N2 = plotdata(filename_bang,filename_bang_avg,tplot, itmax=itmax_inds[2])
ax.semilogy(k2, N2, 'r-', alpha=0.25,  label=labels[2])

filename_bang   = basedir + simres[1] + fft_name
filename_bang_avg   = basedir + simres[1] + avg_name
k1,N1 = plotdata(filename_bang,filename_bang_avg,tplot, itmax=itmax_inds[1])
ax.semilogy(k1, N1, 'r-', alpha=0.5, label=labels[1])

filename_bang   = basedir + simres[0] + fft_name
filename_bang_avg   = basedir + simres[0] + avg_name
#k0,N0 = plotdata(filename_bang,filename_bang_avg,tplot)
k0,N0 = plotdata(filename_bang,filename_bang_avg,tplot, itmax=itmax_inds[0])
ax.semilogy(k0, N0, 'r-', label=labels[0])

#Vertical line from LSA for fastet growing mode
if 'est_kmax' in locals():
    ax.axvline(est_kmax, color='g', label=None)

fig.text(0.15, 0.65, r'$t-t_{{\rm sat}}\sim{}\,{{\rm ns}}$'.format(1.e+9*tplot))
if "x_limits" in locals():
    ax.set_xlim(x_limits)

#ax.legend(loc='upper right', frameon=False)
savename = basedir + savedirname
plt.savefig(savename, bbox_inches="tight")
