# generate 2x3 convergence plots. Run from PAPER/1D
# run from the script directory

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator, ScalarFormatter)


dir_2D_fid = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_fiducial"
dir_2d_128cm = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_128cm"
dir_2D_nx512 = "/global/project/projectdirs/m3018/Emu/PAPER/2D/2D_nx512"

ylabel = r"$\langle N_{ij}\rangle /\mathrm{Tr}(N)$"

#saturation_factor = 0.9
#start_averaging = 10 # times t_saturation

def i_saturation(N):
    R=0
    Nee = N[:,0,0,R]
    i = np.argmax(Nee < saturation_factor*Nee[0])
    return i

def calc_average_single(filename):
    f = h5py.File(filename,"r")
    t = f["t"]
    #isat = i_saturation(f["N"])
    #tsat = t[isat]
    tstart = 4e-9 #tsat * start_averaging
    istart = np.argmax(np.array(t) > tstart)
    n = np.shape(f["N"])[0] - istart
    Navg = np.sum(f["N"][istart:], axis=0) / n
    Nbaravg = np.sum(f["Nbar"][istart:], axis=0) / n
    f.close()
    return Navg, Nbaravg

def get_metrics(dir_list):
    avg_list = []
    stddev_list = []
    avgbar_list = []
    stddevbar_list = []
    for dirname in dir_list:
        # f1, f2, R/I
        NGlobalAvg = np.zeros((3,3,2))
        NGlobalAvg2 = np.zeros((3,3,2))
        NGlobalAvgbar = np.zeros((3,3,2))
        NGlobalAvg2bar = np.zeros((3,3,2))
        sample_files = glob.glob(dirname+"/*/avg_data.h5")
        for s in sample_files:
            Navg,Navgbar = calc_average_single(s)
            NGlobalAvg += Navg
            NGlobalAvg2 += Navg**2
            NGlobalAvgbar += Navgbar
            NGlobalAvg2bar += Navgbar**2
        nsamples = len(sample_files)
        print(dirname+" has "+str(nsamples)+" samples.")
        NGlobalAvg /= nsamples
        NGlobalAvg2 /= nsamples
        NGlobalAvgbar /= nsamples
        NGlobalAvg2bar /= nsamples
        StdDev = np.sqrt(NGlobalAvg2 - NGlobalAvg**2)#/np.sqrt(nsamples-1)
        StdDevbar = np.sqrt(NGlobalAvg2bar - NGlobalAvgbar**2)#/np.sqrt(nsamples-1)
        avg_list.append(NGlobalAvg)
        stddev_list.append(StdDev)
        avgbar_list.append(NGlobalAvgbar)
        stddevbar_list.append(StdDevbar)
    avg_list = np.array(avg_list)
    stddev_list = np.array(stddev_list)
    avgbar_list = np.array(avgbar_list)
    stddevbar_list = np.array(stddevbar_list)
    return avg_list, stddev_list, avgbar_list, stddevbar_list


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

R=0
I=1

def plot_column(x_list,avg_list,avgbar_list, axes, xlabel):
    marker="."
    ax = axes[0]
    ax.plot(x_list, avg_list[:,0,0,R],
            marker=marker, label=r"$N_{ee}$", color="blue", linewidth=1)
    ax.plot(x_list, avg_list[:,1,1,R],
            marker=marker, label=r"$N_{\mu\mu}$", color="red", linewidth=1)
    ax.plot(x_list, avg_list[:,2,2,R],
            marker=marker, label=r"$N_{\tau\tau}$", color="gold", linewidth=1)
    ax.plot(x_list, avgbar_list[:,0,0,R],
            marker=marker, label=r"$\bar{N}_{ee}$", color="blue", linestyle="--", linewidth=2)
    ax.plot(x_list, avgbar_list[:,1,1,R],
            marker=marker, label=r"$\bar{N}_{\mu\mu}$", color="red", linestyle="--", linewidth=2)
    ax.plot(x_list, avgbar_list[:,2,2,R],
            marker=marker, label=r"$\bar{N}_{\tau\tau}$", color="gold", linestyle="--", linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(x_list)
    ax.set_xticklabels([])    
    
    ax = axes[1]
    ax.plot(x_list, avg_list[:,0,1,R],
            marker=marker, label=r"$\mathrm{Re}(N_{e\mu})$", color="purple", linewidth=1)
    ax.plot(x_list, avg_list[:,0,2,R],
            marker=marker, label=r"$\mathrm{Re}(N_{e\tau})$", color="green", linewidth=1)
    ax.plot(x_list, avg_list[:,1,2,R],
            marker=marker, label=r"$\mathrm{Re}(N_{\mu\tau})$", color="orange",linewidth=1)
    ax.plot(x_list, avg_list[:,0,1,I],
            marker=marker, label=r"$\mathrm{Im}(N_{e\mu})$", color="purple", linewidth=1, alpha=0.5)
    ax.plot(x_list, avg_list[:,0,2,I],
            marker=marker, label=r"$\mathrm{Im}(N_{e\tau})$", color="green", linewidth=1, alpha=0.5)
    ax.plot(x_list, avg_list[:,1,2,I],
            marker=marker, label=r"$\mathrm{Im}(N_{\mu\tau})$", color="orange", linewidth=1, alpha=0.5)
    ax.plot(x_list, avgbar_list[:,0,1,R],
            marker=marker, label=r"$\mathrm{Re}(\bar{N}_{e\mu})$", color="purple", linewidth=2, linestyle="--")
    ax.plot(x_list, avgbar_list[:,0,2,R],
            marker=marker, label=r"$\mathrm{Re}(\bar{N}_{e\tau})$", color="green", linewidth=2, linestyle="--")
    ax.plot(x_list, avgbar_list[:,1,2,R],
            marker=marker, label=r"$\mathrm{Re}(\bar{N}_{\mu\tau})$", color="orange", linewidth=2, linestyle="--")
    ax.plot(x_list, -avgbar_list[:,0,1,I],
            marker=marker, label=r"$-\mathrm{Im}(\bar{N}_{e\mu})$", color="purple", linewidth=2, linestyle="--",alpha=0.5)
    ax.plot(x_list, -avgbar_list[:,0,2,I],
    marker=marker, label=r"$-\mathrm{Im}(\bar{N}_{e\tau})$", color="green", linewidth=2, linestyle="--",alpha=0.5)
    ax.plot(x_list, -avgbar_list[:,1,2,I],
            marker=marker, label=r"$-\mathrm{Im}(\bar{N}_{\mu\tau})$", color="orange", linewidth=2, linestyle="--",alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_list)
    ax.set_xticklabels([str(x) for x in x_list],rotation=90)

def scatter_column(x_list,avg_list,avgbar_list, axes,marker,size):
    ax = axes[0]
    ax.scatter(x_list, avg_list[:,0,0,R],
               marker=marker, color="blue", s=size)
    ax.scatter(x_list, avg_list[:,1,1,R],
            marker=marker, color="red", s=size)
    ax.scatter(x_list, avg_list[:,2,2,R],
            marker=marker, color="gold", s=size)
    ax.scatter(x_list, avgbar_list[:,0,0,R],
            marker=marker, color="blue", linestyle="--", s=size)
    ax.scatter(x_list, avgbar_list[:,1,1,R],
            marker=marker, color="red", linestyle="--", s=size)
    ax.scatter(x_list, avgbar_list[:,2,2,R],
            marker=marker, color="gold", linestyle="--", s=size)
    
    ax = axes[1]
    ax.scatter(x_list, avg_list[:,0,1,R],
            marker=marker, color="purple", s=size)
    ax.scatter(x_list, avg_list[:,0,2,R],
            marker=marker, color="green", s=size)
    ax.scatter(x_list, avg_list[:,1,2,R],
            marker=marker, color="orange",s=size)
    ax.scatter(x_list, avg_list[:,0,1,I],
            marker=marker, color="purple", s=size, alpha=0.5)
    ax.scatter(x_list, avg_list[:,0,2,I],
            marker=marker, color="green", s=size, alpha=0.5)
    ax.scatter(x_list, avg_list[:,1,2,I],
            marker=marker, color="orange", s=size, alpha=0.5)
    ax.scatter(x_list, avgbar_list[:,0,1,R],
            marker=marker, color="purple", s=size, linestyle="--")
    ax.scatter(x_list, avgbar_list[:,0,2,R],
            marker=marker, color="green", s=size, linestyle="--")
    ax.scatter(x_list, avgbar_list[:,1,2,R],
            marker=marker, color="orange", s=size, linestyle="--")
    ax.scatter(x_list, -avgbar_list[:,0,1,I],
            marker=marker, color="purple", s=size, linestyle="--",alpha=0.5)
    ax.scatter(x_list, -avgbar_list[:,0,2,I],
               marker=marker, color="green", s=size, linestyle="--",alpha=0.5)
    ax.scatter(x_list, -avgbar_list[:,1,2,I],
               marker=marker, color="orange", s=size, linestyle="--",alpha=0.5)


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
avg_list, stddev_list, avgbar_list, stddevbar_list = get_metrics(dir_list)
plot_column(x_list, avg_list, avgbar_list, axes[:,0], xlabel)

dir_list = ["../2D/2D_fiducial", "../2D/2D_nx512"]
x_list = [256, 512]
avg_list, stddev_list, avgbar_list, stddevbar_list = get_metrics(dir_list)
scatter_column(x_list, avg_list, avgbar_list, axes[:,0],"x",500)


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
            basedir+"1D_512cm",
            basedir+"1D_1024cm"]
x_list = [1,2,4,8,16,32,64,128,256,512,1024]
xlabel = "Domain Size (cm)"
avg_list, stddev_list, avgbar_list, stddevbar_list = get_metrics(dir_list)
plot_column(x_list, avg_list, avgbar_list, axes[:,1], xlabel)

dir_list = ["../2D/2D_fiducial", "../2D/2D_128cm"]
x_list = [64,128]
avg_list, stddev_list, avgbar_list, stddevbar_list = get_metrics(dir_list)
scatter_column(x_list, avg_list, avgbar_list, axes[:,1],"x",500)

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
            basedir+"1D_dir128",
            basedir+"1D_dir256",
            basedir+"1D_dir512"]
x_list = [2,4,8,16,32,64,128,256,512]
xlabel = "Eq. Directions"
avg_list, stddev_list, avgbar_list, stddevbar_list = get_metrics(dir_list)
plot_column(x_list, avg_list, avgbar_list, axes[:,2], xlabel)

dir_list = ["../2D/2D_4dir", "../2D/2D_fiducial", "../2D/2D_32dir"]
x_list = [4,16,32]
avg_list, stddev_list, avgbar_list, stddevbar_list = get_metrics(dir_list)
scatter_column(x_list, avg_list, avgbar_list, axes[:,2],"x",500)

###################
# axis formatting #
###################
for ax in axes[0,:]:
    ax.set_ylim(0,1)
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    
for ax in axes[1,:]:
    ax.set_ylim(-0.035,0.10999)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(MultipleLocator(0.005))

for ax in axes[:,1:].flatten():
    ax.set_yticklabels([])

for ax in axes[:,0]:
    ax.set_ylabel(ylabel)

for ax in axes.flatten():
    ax.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
    ax.xaxis.set_tick_params(which='major', direction='in', bottom=True, top=True)
    ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)

###########
# legends #
###########
axes[0,2].legend(frameon=True, loc='upper center',fontsize=18,
                 ncol=2, handletextpad=0.4, columnspacing=1.2)
axes[1,2].legend(frameon=True, loc='upper center',fontsize=18,
                 ncol=2, handletextpad=0.4, columnspacing=1.2)

###############
# annotations #
###############
axes[0,0].text(64,.15,"64 cm",fontsize=18)
axes[0,0].text(64,.09,"16 eq. directions",fontsize=18)
axes[0,1].text(8,.15,"256 zones",fontsize=18)
axes[0,1].text(8,.09,"16 eq. directions",fontsize=18)
axes[0,2].text(8,.15,"256 zones",fontsize=18)
axes[0,2].text(8,.09,"64cm",fontsize=18)
for ax in axes[0,:]:
    ax.axhline(1./3., color="gray", alpha=0.5, linewidth=2)
for ax in axes[:,0]:
    ax.axvline(256, color="gray", alpha=0.5, linewidth=2)
for ax in axes[:,1]:
    ax.axvline(64, color="gray", alpha=0.5, linewidth=2)
for ax in axes[:,2]:
    ax.axvline(16, color="gray", alpha=0.5, linewidth=2)
axes[0,0].text(4,.35,"1/3",fontsize=18,color="gray")
axes[0,0].text(150,.7,"Fiducial",fontsize=18,color="gray",rotation=90)

fig.align_xlabels(axes)
fig.align_ylabels(axes)
plt.savefig("1D_convergence.pdf", bbox_inches='tight')

