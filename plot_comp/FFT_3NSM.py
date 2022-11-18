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
def plotdata(filename_FFT, filename_avg, t_in, ind_FFT, ind_avg):
    if not os.path.exists(filename_FFT):
        return [0,],[0,]
    
    fftData = h5py.File(filename_FFT,"r")
    t=np.array(fftData[t_str[ind_FFT]])
    k=np.array(fftData[k_str[ind_FFT]])
    #may not need to do the below if ind_FFT=1....
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
    itmax = np.argmax(Nexavg)
    t = t-t[itmax]
    
    # get time closest to t
    dt = np.abs(t-t_in)
    it = np.argmin(dt)
    #probably a bug here: Nex -> Nxx
    #trace = Nee[it,0]+Nex[it,0]
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


emu_2f_pre = "/global/project/projectdirs/m3761/FLASH/Emu/"

#emu_2f_sims = ['NSM_1/']
emu_2f_sims = ['NSM_1/', 'NSM_2/32dir/', 'NSM_3/32dir/']
#emu_2f_res = ['merger_2F/', 'merger_2F_lowres/']
emu_2f_res = ['merger_2F/']


emu_3f_pre = "/global/project/projectdirs/m3761/FLASH/Emu/"

emu_3f_sims = ['NSM_1/']
#emu_3f_res = ['merger_3F/', 'merger_3F_lowres/']
emu_3f_res = ['merger_3F/']


bang_pre = "/global/project/projectdirs/m3761/FLASH/FFI_3D/"

bang_sims = ['NSM_1/', 'NSM_2/t3/', 'NSM_3/t6/']
#bang_res = ['sim/', 'res_a/', 'res_b/']
bang_res = ['sim/']


test_fig_labels = [r'${\rm NSM}\,1$', r'${\rm NSM}\,2$', r'${\rm NSM}\,3$']

h5_filename_old = "reduced_data_old.h5"
h5_filename_orig = "reduced_data.h5"
h5_filename_emu = "plt_reduced_data.h5"

h5_pow_filename_orig = "reduced_data_fft_power.h5"
h5_pow_filename_emu = "plt_reduced_data_fft_power.h5"

tplot = -0.1e-9

for i in range(3):

    ax = axes[i]


    #############
    # plot data #
    #############
    for j in range(1):

        if i == 0:
            if j < 2:
                h5_filename = h5_filename_emu
                ind_avg = 1
                h5_pow_filename = h5_pow_filename_emu
                ind_pow = 1
                if i == 0 and j == 0:
                    h5_pow_filename = h5_pow_filename_orig
                    ind_pow = 0
                if i == 0 and j == 1:
                    h5_filename = h5_filename_orig
                    ind_avg = 0
                    h5_pow_filename = h5_pow_filename_orig
                    ind_pow = 0
                filename_emu_2f_avg = emu_2f_pre + emu_2f_sims[i] + emu_2f_res[j] + h5_filename
                filename_emu_2f_pow = emu_2f_pre + emu_2f_sims[i] + emu_2f_res[j] + h5_pow_filename
                k1,N1 = plotdata(filename_emu_2f_pow,filename_emu_2f_avg,tplot,ind_pow,ind_avg)
                ax.semilogy(k1, N1, 'k-', label=r'${\rm {\tt EMU}\,\,(2f)}$')

        if i == 0 and j < 2:
            if j == 0:
                h5_filename = h5_filename_orig
                ind_avg = 0
                h5_pow_filename = h5_pow_filename_orig
                ind_pow = 0
            else:
                h5_filename = h5_filename_orig
                ind_avg = 0
                h5_pow_filename = h5_pow_filename_orig
                ind_pow = 0
            filename_emu_3f_avg = emu_3f_pre + emu_3f_sims[i] + emu_3f_res[j] + h5_filename
            filename_emu_3f_pow = emu_3f_pre + emu_3f_sims[i] + emu_3f_res[j] + h5_pow_filename
            k2,N2 = plotdata(filename_emu_3f_pow,filename_emu_3f_avg,tplot,ind_pow,ind_avg)
            ax.semilogy(k2, N2, 'k--', label=r'${\rm {\tt EMU}\,\,(3f)}$')

        h5_filename = h5_filename_orig
        ind_avg = 0
        h5_pow_filename = h5_pow_filename_orig
        ind_pow = 0
        #special cases:
        if i == 0 and j == 2:
            filename_bang_avg = bang_pre + bang_sims[i] + 'res_c/' + h5_filename
            filename_bang_pow = bang_pre + bang_sims[i] + 'res_c/' + h5_pow_filename
        else:
            filename_bang_avg = bang_pre + bang_sims[i] + bang_res[j] + h5_filename
            filename_bang_pow = bang_pre + bang_sims[i] + bang_res[j] + h5_pow_filename
        k3,N3 = plotdata(filename_bang_pow,filename_bang_avg,tplot,ind_pow,ind_avg)
        ax.semilogy(k3, N3, 'r-', label=r'${\rm {\tt FLASH}\,\,(2f)}$')

    if i == 0:
        ax.set_ylabel(r"$\langle N_{ee}(t)/N_{ee}(0)\rangle$")
        #ytick_vals = [1.e-7, 1.e-5, 1.e-3, 1.e-1]
        #ytick_labs = [r'$10^{{{}}}$'.format(num) for num in [-7, -5, -3, -1]]
        ax.legend(loc='upper right', frameon=False)
    else:
        plt.setp(ax.get_yticklabels(), visible=False)

    ##############
    # formatting #
    ##############
    #ax.axhline(N_fe, color="green")
    #ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()
    ax.set_title(r'${{\rm NSM}}\,\,{}$'.format(i+1))

#Vertical line from LSA for fastet growing mode
#ax.axvline(5.64, color='g', label=None)

plt.savefig("3NSM_N_ex_FFT.pdf", bbox_inches="tight")

#ind1 = np.argmax(N1)
#print('emu_2f', ind1, k1[ind1], N1[ind1])
#ind2 = np.argmax(N2)
#print('emu_3f', ind2, k2[ind2], N2[ind2])
ind3 = np.argmax(N3)
print('flash', ind3, k3[ind3], N3[ind3])
