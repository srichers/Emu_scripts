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
ax.set_ylabel(r"$\mathcal{D}(k)$")
ax.set_xlim(-5,50)
#ax.set_ylim(1.e-20,1.0)

#############
# plot data #
#############
#filename_1 = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/sim1/reduced_data.h5"
#filenames_avg = ["/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert4/reduced_data.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert3/reduced_data.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert1/reduced_data.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert2/reduced_data.h5"]
#filenames_fft = ["/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert4/reduced_data_fft_power.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert3/reduced_data_fft_power.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert1/reduced_data_fft_power.h5", \
#    "/pscratch/sd/e/egrohs/FFI_3D/nuMixing/NSM_1/w_hv/diag_pert2/reduced_data_fft_power.h5"]
#tdiff = []

#filenames_avg = ["./nuM/sim1/reduced_data.h5", "./MPC/sim1/reduced_data.h5"]
#filenames_fft = ["./nuM/sim1/reduced_data_fft_power.h5", "./MPC/sim1/reduced_data_fft_power.h5"]
#labels = [r'$\nu{\rm M}$', r"${\rm MPC}$"]
#tdiff = -0.02e-9

#NSM_1 w/ & w/o H_M:
#filenames_avg = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t2/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/sim/reduced_data.h5"]
#filenames_fft = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t2/sim/reduced_data_fft_power.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/sim/reduced_data_fft_power.h5"]
#tdiff = -0.02e-9
#labels = [r'$H_M=0$', r"$H_M\ne0$"]
#namestr = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/comp_t/Nex_FFT_2comp.pdf"

#NSM_3 different nblocks:
#filenames_avg = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t2/res_a/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/res_a_xmax_large/reduced_data.h5"]
#filenames_fft = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t2/res_a/reduced_data_fft_power.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/t3/res_a_xmax_large/reduced_data_fft_power.h5"]
#tdiff = -0.05e-9
#labels = [r'$N_{x,y}=16$', r"$N_{x,y}=1$"]
#namestr = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_3/MPC/d_pert/comp_t23/Nex_FFT_2comp_xmax_large.pdf"

#Beam in FFI_1D/changing_N_nuebar:
filenames_avg = ["sim" + str((i+1)*0.2)[0:3] + "/reduced_data.h5" for i in range(5)]
filenames_fft = ["sim" + str((i+1)*0.2)[0:3] + "/reduced_data_fft_power.h5" for i in range(5)]
labels = [r"$\overline{{N}}_{{ee}}/N_{{ee}} = {:.1f}$".format((i+1)*0.2) for i in range(5)]
tdiff = -0.025e-9
namestr = "./comp_res/Nex_FFT_mult_comp.pdf"

##FFT vs. FFTz for NSM_2.5/d_pert/t2/sim_xy_large:
#filenames_avg = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data.h5", \
#        "/pscratch/sd/e/egrohs/FFI_3D/MPC/NSM/NSM_2.5/d_pert/t2/sim_xy_large/reduced_data.h5"]
#filenames_fft = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_2.5/d_pert/t2/xy_large/sim/reduced_data_fft_power.h5", \
#        "/pscratch/sd/e/egrohs/FFI_3D/MPC/NSM/NSM_2.5/d_pert/t2/sim_xy_large/reduced_data_fftz_power.h5"]
#namestr = "./Nex_FFT_mult_comp.pdf"
#labels = [r"${\rm 3D}$", r"$z{\rm -dir}$"]
#tdiff = -0.05e-9

lstyle = ['-', '--', '-.', ':']
lcolor = ['r', 'b', 'g', 'k', 'm']

for i,filename_a in enumerate(filenames_avg):
    filename_f = filenames_fft[i]
    #tplot = tdiff[i]
    tplot = tdiff
    k,N = plotdata(filename_f,filename_a,tplot)
    style_ind = i % len(lstyle)
    color_ind = i % len(lcolor)
    Nmax = np.max(N)
    ax.semilogy(k, N/Nmax, linestyle=lstyle[style_ind], color=lcolor[color_ind],  label=labels[i])

#ax.legend(loc='upper right', frameon=False)
ax.legend(loc='upper right', fontsize=12, frameon=False)
plt.savefig(namestr, bbox_inches="tight")
