# generate 2x3 convergence plots. Run from PAPER/1D
# run from the script directory

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator, ScalarFormatter)

Nex_grow_start = 1e-4
Nex_grow_stop = 1e-2
Fex_grow_tstart = .15e-9
Fex_grow_tstop = .22e-9
tstart_nonlinear = 3e-9

R=0
I=1

def calc_average_single(filename):
    print(filename)
    f = h5py.File(filename,"r")
    t = f["t"]
    istart = np.argmax(np.array(t) > tstart_nonlinear)
    n = np.shape(f["N_avg_mag"])[0] - istart
    Navg = np.sum(f["N_avg_mag"][istart:], axis=0) / n

    Noffdiag = f["N_avg_mag"][:,0,1]
    Foffdiag = f["F_avg_mag"][:,0,0,1]

    past_start = np.where(Noffdiag > Nex_grow_start)[0]
    past_stop  = np.where(Noffdiag > Nex_grow_stop )[0]
    growth_rate = 0
    Fgrowth_rate = 0
    if(len(past_stop)>0):
        istart = past_start[0]
        istop = past_stop[0]
        growth_rate = np.log(Noffdiag[istop]/Noffdiag[istart]) / (t[istop]-t[istart])
        Fgrowth_rate = np.log(Foffdiag[istop]/Foffdiag[istart]) / (t[istop]-t[istart])

    f.close()

    return Navg, growth_rate/1e10, Fgrowth_rate/1e10

def get_metrics(dir_list):
    avg_list = []
    growth_rate_list = []
    Fgrowth_rate_list = []
    for dirname in dir_list:
        filename = dirname+"/reduced_data.h5"
        Navg,growth_rate, Fgrowth_rate = calc_average_single(filename)
        avg_list.append(Navg)
        growth_rate_list.append(growth_rate)
        Fgrowth_rate_list.append(Fgrowth_rate)

    return np.array(avg_list), np.array(growth_rate_list), np.array(Fgrowth_rate_list)


####################
# create the plots #
####################
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
fig, axes = plt.subplots(2,3, figsize=(15,10))
plt.subplots_adjust(hspace=0,wspace=0.0)


def plot_column(x_list,avg_list,growth_rate_list, Fgrowth_rate_list,axes, xlabel):
    marker="."
    ax = axes[0]
    ax.plot(x_list, avg_list[:,0,0],
            marker=marker, label=r"$N_{ee}$", color="black", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(x_list)
    ax.set_xticklabels([])    
    
    ax = axes[1]
    ax.plot(x_list, growth_rate_list, color="blue", marker=marker, linewidth=1)
    ax.plot(x_list, Fgrowth_rate_list, color="red", marker=marker, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_list)
    ax.set_xticklabels([str(x) for x in x_list],rotation=90)

def scatter_column(x_list,avg_list,axes,marker,size):
    ax = axes[0]
    ax.scatter(x_list, avg_list[:,0,0],
               marker=marker, color="blue", s=size)
    
    ax = axes[1]


##############
# GRID ZONES #
##############
basedir="../1D/converge_nx/"
dir_list = [basedir+"1D_nx4",
            basedir+"1D_nx8",
            basedir+"1D_nx16",
            basedir+"1D_nx32",
            basedir+"1D_nx64",
            basedir+"1D_nx128",
            basedir+"1D_nx256",
            basedir+"1D_nx512",
            basedir+"1D_nx1024",
            basedir+"1D_nx2048"]
x_list = [4,8,16,32,64,128,256,512,1024,2048]
xlabel = "Grid Zones"
avg_list, growth_rate_list, Fgrowth_rate_list = get_metrics(dir_list)
plot_column(x_list, avg_list, growth_rate_list, Fgrowth_rate_list, axes[:,0], xlabel)

dir_list = ["../2D/2D_fiducial", "../2D/2D_nx512"]
x_list = [256, 512]
#avg_list, growth_rate_list = get_metrics(dir_list)
#scatter_column(x_list, avg_list, axes[:,0],"x",500)


###############
# DOMAIN SIZE #
###############
basedir="../1D/converge_domain/"
dir_list = [basedir+"1D_1cm",
            basedir+"1D_2cm",
            basedir+"1D_4cm",
            basedir+"1D_8cm",
            basedir+"1D_16cm",
            basedir+"1D_32cm",
            basedir+"1D_64cm",
            basedir+"1D_128cm",
            basedir+"1D_256cm",
            basedir+"1D_512cm"]#,
            #basedir+"1D_1024cm"]
x_list = [1,2,4,8,16,32,64,128,256,512]#,512,1024]
xlabel = "Domain Size (cm)"
avg_list, growth_rate_list, Fgrowth_rate_list = get_metrics(dir_list)
plot_column(x_list, avg_list, growth_rate_list, Fgrowth_rate_list, axes[:,1], xlabel)

dir_list = ["../2D/2D_fiducial", "../2D/2D_128cm"]
x_list = [64,128]
#avg_list, growth_rate_list = get_metrics(dir_list)
#scatter_column(x_list, avg_list, axes[:,1],"x",500)

##############
# DIRECTIONS #
##############
basedir="../1D/converge_direction/"
dir_list = [basedir+"1D_dir2",
            basedir+"1D_dir4",
            basedir+"1D_dir8",
            basedir+"1D_dir16",
            basedir+"1D_dir32",
            basedir+"1D_dir64",
            basedir+"1D_dir128"]#,
            #basedir+"1D_dir256",
            #basedir+"1D_dir512"]
x_list = [2,4,8,16,32,64,128]#,256,512]
xlabel = "Eq. Directions"
avg_list, growth_rate_list, Fgrowth_rate_list = get_metrics(dir_list)
plot_column(x_list, avg_list, growth_rate_list, Fgrowth_rate_list, axes[:,2], xlabel)

dir_list = ["../2D/2D_4dir", "../2D/2D_fiducial", "../2D/2D_32dir"]
x_list = [4,16,32]
#avg_list, growth_rate_list = get_metrics(dir_list)
#scatter_column(x_list, avg_list, axes[:,2],"x",500)

###################
# axis formatting #
###################
for ax in axes[0,:]:
    ax.set_ylim(.25,1.05)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
for ax in axes[1,:]:
    ax.set_ylim(-0.2,7)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(.2))

for ax in axes[:,1:].flatten():
    ax.set_yticklabels([])

axes[0,0].set_ylabel(r"$\langle N_{ee}\rangle /\mathrm{Tr}(N)$")
axes[1,0].set_ylabel(r"$\mathrm{Im}(\omega)\,(10^{10}\,\mathrm{s}^{-1})$")

for ax in axes.flatten():
    ax.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
    ax.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True)
    ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)

###########
# legends #
###########
#axes[0,2].legend(frameon=True, loc='upper center',fontsize=18,
#                 ncol=2, handletextpad=0.4, columnspacing=1.2)
#axes[1,2].legend(frameon=True, loc='upper center',fontsize=18,
#                 ncol=2, handletextpad=0.4, columnspacing=1.2)

###############
# annotations #
###############
y1 = .7
y2 = .63
axes[0,0].text(32,y1,"64 cm",fontsize=18)
axes[0,0].text(32,y2,"16 eq. directions",fontsize=18)
axes[0,1].text(3,y1,"16 zones/cm",fontsize=18)
axes[0,1].text(3,y2,"16 eq. directions",fontsize=18)
axes[0,2].text(20,y1,"16 zones/cm",fontsize=18)
axes[0,2].text(20,y2,"64 cm",fontsize=18)
for ax in axes[0,:]:
    ax.axhline(1./3., color="gray", alpha=0.5, linewidth=2)
for ax in axes[1,:]:
    ax.axhline(6.5, color="gray", alpha=0.5, linewidth=2)
    ax.axhline(1.06, color="gray", alpha=0.5, linewidth=2)
for ax in axes[:,0]:
    ax.axvline(1024, color="gray", alpha=0.5, linewidth=2)
for ax in axes[:,1]:
    ax.axvline(64, color="gray", alpha=0.5, linewidth=2)
for ax in axes[:,2]:
    ax.axvline(16, color="gray", alpha=0.5, linewidth=2)
axes[0,0].text(4,.35,"1/3",fontsize=18,color="gray")
axes[1,0].text(4,6,r"$6.50\times10^{10}\,\mathrm{s}^{-1}$",fontsize=18,color="gray")
axes[1,1].text(2,1.2,r"$1.06\times10^{10}\,\mathrm{s}^{-1}$",fontsize=18,color="gray")
axes[0,0].text(600,.8,"Fiducial",fontsize=18,color="gray",rotation=90)
axes[0,1].text(40,.8,"Fiducial",fontsize=18,color="gray",rotation=90)
axes[0,2].text(12,.8,"Fiducial",fontsize=18,color="gray",rotation=90)

fig.align_xlabels(axes)
fig.align_ylabels(axes)
plt.savefig("1D_convergence.pdf", bbox_inches='tight')

