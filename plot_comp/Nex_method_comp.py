# Run from /ocean/projects/phy200048p/shared to generate plot showing time evolution of <fee> at different dimensionalities

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

    
t_str = ["t", "t(s)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]

######################
# read averaged data #
######################
def plotdata(filename,a,b,ind):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData[t_str[ind]])*1e9
    N=np.array(avgData[N_str[ind]])[:,a,b]
    avgData.close()
    return t, N

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
mpl.rcParams['text.usetex'] = True
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


fig, axes = plt.subplots(1,1, figsize=(6,5))

#comp of methods in NSM1
#filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/MPC/d_pert/t3/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/sim/reduced_data.h5", \
#        "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_1/merger_2F/plt_reduced_data.h5"]
#ind = np.zeros([3], dtype=np.int8)
#ind[2] = 1
#lstyle = ['-', '-.', '--', ':']
#lcolor = ['r', 'b', 'k', 'g', 'm']
#labels = [r"${\rm {\tt FLASH}}$", r"${\rm {\tt FLASH}}\,\,(ri)$", r"${\rm {\tt Emu}}$"]
#xlimits = (-0.3, 0.3) #ns
##LSA growth rate:
#est_imo = 7.25e10 #s^{-1}
#Nex_pow = -3.0
#scale_fact = 5.0
##ind1 = 205
#ind_offset = 5
#savedir = "global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_1/method_comp/"

#comp of methods in NSM2.5
filenames = ["/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_4/d_pert/t2/xy_large/sim/reduced_data.h5", \
    "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_4/od_pert/t2/xy_large/sim/reduced_data.h5", \
    "/global/cfs/projectdirs/m3761/FLASH/Emu/NSM_4/1res/correct_Nxx/NSM4_matchevan_correctNxx_long_diagonalpert/plt_reduced_data.h5"]
ind = np.zeros([3], dtype=np.int8)
ind[2] = 1
labels = [r"${\rm {\tt FLASH}}\,(\delta N_{cc})$", r"${\rm {\tt FLASH}}\,(\delta N_{ab})$", r"${\rm {\tt Emu}}\,(\delta N_{cc})$"]
lstyle = ['-', '-.', '--', ':']
lcolor = ['r', 'm', 'k', 'g', 'b']
xlimits = (-1.3, 0.2) #ns
##from LSA:
est_imo = 1.39e10 #s^{-1}
Nex_pow = -2.2
scale_fact = 10.0
ind1 = 60
ind_offset = 8
legend_fontsize = 19
ax_title = r'${\rm NSM}4$'
savedir = "/global/cfs/projectdirs/m3761/FLASH/FFI_3D/NSM_4/comp_pert/"

for i,filename in enumerate(filenames):
    t,Nee = plotdata(filename,0,0,ind[i])
    tex,Nex = plotdata(filename,0,1,ind[i])
    txx,Nxx = plotdata(filename,1,1,ind[i])
    n_2F = Nee[0] + Nxx[0]
    tmax = t[np.argmax(Nex)]
    if (i == 0) and ("est_imo" in locals()):
        est_imo_ns = est_imo/1.e+9 #ns^{-1}
        Nex_base = 10.0**(Nex_pow)
        if "ind1" not in locals():
            ind1 = np.argmin(np.abs(np.log(Nex[1:]/Nex_base)))
        ind2 = ind1 + ind_offset
        t_line = [t[ind1], t[ind2]]
        N_line = [scale_fact*Nex[ind1], scale_fact*Nex[ind1]*np.exp(est_imo_ns*(t[ind2] - t[ind1]))]
        print(scale_fact*Nex[ind1])
        axes.semilogy(t_line-tmax, N_line, color='orange', label=None)
    #tmax = 0.0
    style_ind = i % len(lstyle)
    color_ind = i % len(lcolor)
    n_2F_eq = n_2F/2.0
    axes.semilogy(t-tmax, Nex/n_2F, label=labels[i], linestyle=lstyle[style_ind], color=lcolor[color_ind])


##############
# formatting #
##############
axes.set_xlabel(r'$t-t_{\rm sat}\,(10^{-9}\,{\rm s})$')
axes.tick_params(axis='both', which='both', direction='in', right=True,top=True)
axes.xaxis.set_minor_locator(AutoMinorLocator())
axes.yaxis.set_minor_locator(AutoMinorLocator())
axes.minorticks_on()
if "xlimits" in locals():
    axes.set_xlim(xlimits)
else:
    axes.set_xlim(-0.4, 0.8)
axes.set_ylabel(r'$\langle|N_{ex}|\rangle/\langle{\rm Tr}[N]\rangle$')
axes.set_title(ax_title)

if "legend_fontsize" in locals():
    axes.legend(loc='best', fontsize=legend_fontsize, frameon=False)
else:
    axes.legend(loc='best', frameon=False)
plt.savefig(savedir + "Nex_method_comp.pdf", bbox_inches="tight")
