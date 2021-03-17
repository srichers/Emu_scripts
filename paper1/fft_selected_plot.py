import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator, ScalarFormatter)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str,
                    default="/global/project/projectdirs/m3018/Emu/PAPER/1D/converge_nx/1D_nx4096/reduced_data_fft.h5",
                    help="Path to the reduced data HDF5 file containing FFT data for a simulation timeseries.")
parser.add_argument("-o", "--output", type=str,
                    default="fft_selected_plot",
                    help="Prefix for the output plotfile names.")
parser.add_argument("-phase", "--phase", action="store_true", help="Plot FFT phase as well as magnitude.")
args = parser.parse_args()

cmap=mpl.cm.jet

f = h5py.File(args.data,"r")
#f = h5py.File("/global/project/projectdirs/m3018/Emu/PAPER/1D/converge_nx/1D_nx4096/reduced_data_fft.h5","r")
#f = h5py.File("/global/project/projectdirs/m3018/Emu/PAPER/1D/converge_domain/1D_512cm/reduced_data_fft.h5","r")

t        = np.array(f["t"])       
kz       = np.array(f["kz"])*2.*np.pi      

N00_FFT  = np.array(f["N00_FFT"]) 
N11_FFT  = np.array(f["N11_FFT"]) 
N22_FFT  = np.array(f["N22_FFT"]) 
N01_FFT  = np.array(f["N01_FFT"]) 
N02_FFT  = np.array(f["N02_FFT"]) 
N12_FFT  = np.array(f["N12_FFT"]) 
Fx00_FFT = np.array(f["Fx00_FFT"])
Fx11_FFT = np.array(f["Fx11_FFT"])
Fx22_FFT = np.array(f["Fx22_FFT"])
Fx01_FFT = np.array(f["Fx01_FFT"])
Fx02_FFT = np.array(f["Fx02_FFT"])
Fx12_FFT = np.array(f["Fx12_FFT"])

if args.phase:
    N00_FFT_phase  = np.array(f["N00_FFT_phase"]) 
    N11_FFT_phase  = np.array(f["N11_FFT_phase"]) 
    N22_FFT_phase  = np.array(f["N22_FFT_phase"]) 
    N01_FFT_phase  = np.array(f["N01_FFT_phase"]) 
    N02_FFT_phase  = np.array(f["N02_FFT_phase"]) 
    N12_FFT_phase  = np.array(f["N12_FFT_phase"]) 
    Fx00_FFT_phase = np.array(f["Fx00_FFT_phase"])
    Fx11_FFT_phase = np.array(f["Fx11_FFT_phase"])
    Fx22_FFT_phase = np.array(f["Fx22_FFT_phase"])
    Fx01_FFT_phase = np.array(f["Fx01_FFT_phase"])
    Fx02_FFT_phase = np.array(f["Fx02_FFT_phase"])
    Fx12_FFT_phase = np.array(f["Fx12_FFT_phase"])

f.close()

print(np.shape(N00_FFT))

####################
# definitions      #
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

def rebin(a):
    for iter in range(3):
        a = np.array([ (a[2*i]+a[2*i+1])/2. for i in range(int(len(a)/2))])
    return a

nblocks = 50
dt = 0.1e-9 #t[-1]/nblocks
ibref=9
def makeplot(ax, Q, log=True):
    if log:
        ax.set_yscale('log')
    for ib in range(nblocks):
        locs = np.where((t>dt*ib) & (t<=dt*(ib+1)))
        navg = len(t[locs])
        Q_avg = np.squeeze(np.sum(Q[locs,:],axis=1)/navg)
        if(ib==1):
            Q0 = Q_avg
        #Q_avg_bin = rebin(Q_avg)
        #kz_bin = rebin(kz)
        else:
            # (ib+1)/nblocks sets the color to the final time in each block
            ax.plot(kz/np.sqrt((ib+1)/ibref), Q_avg, color=cmap((ib+1.0)/nblocks), linewidth=2) #
    #ax.plot(kz/np.sqrt((ib+1)/ibref), Q_avg, color="black", linewidth=2) #
    return Q0

##############################
# create the magnitude plots #
##############################

fig, axes = plt.subplots(3,1, figsize=(6,12))
plt.subplots_adjust(hspace=0,wspace=0.0)

print((ibref+1)*dt)
ib=1
color="salmon" #cmap(ib/nblocks)
Q0 = makeplot(axes[0], N00_FFT)
axes[0].plot(kz/np.sqrt((ib+1)/ibref),Q0, color=color, linewidth=2)
Q0 = makeplot(axes[1], N01_FFT)
axes[1].plot(kz/np.sqrt((ib+1)/ibref),Q0, color=color, linewidth=2)
Q0 = makeplot(axes[2], Fx01_FFT)
axes[2].plot(kz/np.sqrt((ib+1)/ibref),Q0, color=color, linewidth=2)

# time labels
axes[0].text(-39,3e21,r"$\mathbf{0-0.1\,\mathrm{ns}}$",color="blue",fontsize=12)
axes[0].text(-39,7e23,r"$\mathbf{0.1-0.2\,\mathrm{ns}}$",color="salmon",fontsize=12)
axes[0].text(-30,3e25,r"$\mathbf{0.2-0.3\,\mathrm{ns}}$",color="blue",fontsize=12,rotation=35)
axes[1].text(2,1e26,r"$\mathbf{0-0.1\,\mathrm{ns}}$",color="blue",fontsize=12)
axes[1].text(-4,1e27,r"$\mathbf{0.1-0.2\,\mathrm{ns}}$",color="salmon",fontsize=12, rotation=80)
axes[1].text(-18,1e27,r"$\mathbf{0.2-0.3\,\mathrm{ns}}$",color="blue",fontsize=12,rotation=55)


ax = fig.add_axes([0,0,0,0])
a = np.array([[0,5]])
img = plt.imshow(a,cmap=cmap, vmin=0, vmax=5)
plt.gca().set_visible(False)
cax = fig.add_axes([0.125, .89, .775, 0.02])
cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
cbar = plt.colorbar(cax=cax,orientation='horizontal')
cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
cax.minorticks_on()
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_label_position('top')
cax.xaxis.set_minor_locator(MultipleLocator(0.1))
cax.axvspan(0.1,0.2,color=color)
#cax.axvspan(4.9,5.0,color="black")

for ax in axes:
    ax.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
    ax.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=True)
    ax.set_xlim(-40,40)
    ax.minorticks_on()
    
for ax in axes[0:2]:
    ax.set_xticklabels([])
    
axes[2].set_xlabel(r"$k\sqrt{\frac{1\,\mathrm{ns}}{t}}\,(\mathrm{cm}^{-1})$")
    
axes[0].set_ylim(1.01e20, 1e34)
axes[1].set_ylim(1e25, 1e34)
axes[2].set_ylim(1e25, 3e28)


axes[0].set_ylabel(r"$|\widetilde{n}_{ee}|\,(\mathrm{cm}^{-2})$")
axes[1].set_ylabel(r"$|\widetilde{n}_{e\mu}|\,(\mathrm{cm}^{-2})$")
axes[2].set_ylabel(r"$|\widetilde{\mathbf{f}}_{e\mu}^{(x)}|\,(\mathrm{cm}^{-2})$")

#axes[0].axvline(2.*np.pi/2.20/np.sqrt((1+1)/ibref), color="blue", linewidth=2)
#axes[1].axvline(2.*np.pi/2.20/np.sqrt((1+1)/ibref), color="blue", linewidth=2)
#axes[2].axvline(2.*np.pi/4.45/np.sqrt((1+1)/ibref), color="red", linewidth=2)

plt.savefig("{}_magnitude.png".format(args.output), bbox_inches='tight',dpi=300)
plt.savefig("{}_magnitude.pdf".format(args.output), bbox_inches='tight',dpi=300)

if args.phase:
    ##############################
    # create the phase plots     #
    ##############################

    fig, axes = plt.subplots(3,1, figsize=(6,12))
    plt.subplots_adjust(hspace=0,wspace=0.0)

    print((ibref+1)*dt)
    ib=1
    color="salmon" #cmap(ib/nblocks)
    Q0 = makeplot(axes[0], N00_FFT_phase, log=False)
    axes[0].plot(kz/np.sqrt((ib+1)/ibref),Q0, color=color, linewidth=2)
    Q0 = makeplot(axes[1], N01_FFT_phase, log=False)
    axes[1].plot(kz/np.sqrt((ib+1)/ibref),Q0, color=color, linewidth=2)
    Q0 = makeplot(axes[2], Fx01_FFT_phase, log=False)
    axes[2].plot(kz/np.sqrt((ib+1)/ibref),Q0, color=color, linewidth=2)

    ax = fig.add_axes([0,0,0,0])
    a = np.array([[0,5]])
    img = plt.imshow(a,cmap=cmap, vmin=0, vmax=5)
    plt.gca().set_visible(False)
    cax = fig.add_axes([0.125, .89, .775, 0.02])
    cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
    cbar = plt.colorbar(cax=cax,orientation='horizontal')
    cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
    cax.minorticks_on()
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_minor_locator(MultipleLocator(0.1))
    cax.axvspan(0.1,0.2,color=color)
    #cax.axvspan(4.9,5.0,color="black")

    for ax in axes:
        ax.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
        ax.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=True)
        ax.set_xlim(-40,40)
        ax.minorticks_on()
        
    for ax in axes[0:2]:
        ax.set_xticklabels([])
        
    axes[2].set_xlabel(r"$k\sqrt{\frac{1\,\mathrm{ns}}{t}}\,(\mathrm{cm}^{-1})$")
        
    axes[0].set_ylim(-np.pi, np.pi)
    axes[1].set_ylim(-np.pi, np.pi)
    axes[2].set_ylim(-np.pi, np.pi)


    axes[0].set_ylabel(r"$\phi(\widetilde{n}_{ee})$")
    axes[1].set_ylabel(r"$\phi(\widetilde{n}_{e\mu})$")
    axes[2].set_ylabel(r"$\phi(\widetilde{\mathbf{f}}_{e\mu}^{(x)})$")

    #axes[0].axvline(2.*np.pi/2.20/np.sqrt((1+1)/ibref), color="blue", linewidth=2)
    #axes[1].axvline(2.*np.pi/2.20/np.sqrt((1+1)/ibref), color="blue", linewidth=2)
    #axes[2].axvline(2.*np.pi/4.45/np.sqrt((1+1)/ibref), color="red", linewidth=2)

    axes[0].text(-40,1e22,"$t=0-0.1\,\mathrm{ns}$",color="blue")

    plt.savefig("{}_phase.png".format(args.output), bbox_inches='tight',dpi=300)
    plt.savefig("{}_phase.pdf".format(args.output), bbox_inches='tight',dpi=300)
