import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator, ScalarFormatter)

cmap=mpl.cm.plasma

f = h5py.File("/global/project/projectdirs/m3018/Emu/PAPER/1D/converge_domain/1D_512cm/reduced_data_fft.h5","r")
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
f.close()

print(np.shape(N00_FFT))

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
fig, axes = plt.subplots(4,3, figsize=(30,30))
plt.subplots_adjust(hspace=0,wspace=0.0)

def rebin(a):
    for iter in range(3):
        a = np.array([ (a[2*i]+a[2*i+1])/2. for i in range(int(len(a)/2))])
    return a

nblocks = 50
dt = 0.1e-9 #t[-1]/nblocks
def makeplot(ax, Q):
    ax.set_yscale('log')
    for ib in range(nblocks):
        locs = np.where((t>dt*ib) & (t<=dt*(ib+1)))
        navg = len(t[locs])
        Q_avg = np.squeeze(np.sum(Q[locs,:],axis=1)/navg)
        #Q_avg_bin = rebin(Q_avg)
        #kz_bin = rebin(kz)
        ax.plot(kz, Q_avg, color=cmap(ib/nblocks))
    ax.plot(kz, Q_avg, color="black")

makeplot(axes[0,0], N00_FFT)
makeplot(axes[0,1], N11_FFT)
makeplot(axes[0,2], N22_FFT)
makeplot(axes[1,0], N01_FFT)
makeplot(axes[1,1], N02_FFT)
makeplot(axes[1,2], N12_FFT)
makeplot(axes[2,0], Fx00_FFT)
makeplot(axes[2,1], Fx11_FFT)
makeplot(axes[2,2], Fx22_FFT)
makeplot(axes[3,0], Fx01_FFT)
makeplot(axes[3,1], Fx02_FFT)
makeplot(axes[3,2], Fx12_FFT)

for ax in axes.flatten():
    ax.set_ylim(1.01e20, 1e34)
    ax.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
    ax.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=True)
    ax.set_xlim(-10,9.99)

for ax in axes[:,2]:
    ax.set_xlim(-10,10)
    
for ax in axes[0:3,:].flatten():
    ax.set_xticklabels([])
    
for ax in axes[3,:].flatten():
    ax.set_xlabel(r"$k\,(\mathrm{cm}^{-1})$")
    
for ax in axes[:,1:].flatten():
    ax.set_yticklabels([])

for ax in axes[0:2,0].flatten():
    ax.set_ylabel(r"$\widetilde{N}\,(\mathrm{cm}^{-2})$")
for ax in axes[2:4,0].flatten():
    ax.set_ylabel(r"$\widetilde{F}^{(x)}\,(\mathrm{cm}^{-2})$")

for ax in axes[0:2,:].flatten():
    ax.axvline(2.*np.pi/2.20, color="blue", linewidth=2)

for ax in axes[2:4,:].flatten():
    ax.axvline(2.*np.pi/4.45, color="red", linewidth=2)

plt.savefig("fft_average_plot.pdf", bbox_inches='tight')

