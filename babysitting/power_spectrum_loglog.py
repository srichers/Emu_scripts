import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#variables=["N","Fx","Fy","Fz"]
#flavors=["00","11","22","01","02","12"]
variables=["N"]
flavors=["00","11","01"]
cmap=mpl.cm.jet

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
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


def makeplot(v,f,data):
    plt.close('all')
    fig, ax = plt.subplots(1,1, figsize=(8,6))

    # get appropriate data
    t=data["t"]
    k=data["k"]
    fft = data[v+f+"_FFT"]
    fft = np.array(fft)
    #total_power = np.sum(fft)
    for it in range(len(t)):
        total_power = np.sum(fft[it,:])
        ax.loglog(k[:], fft[it,:-1], color=cmap(t[it]/t[-1]))
    
    # colorbar
    cax = fig.add_axes([0.125, .89, .775, 0.02])
    cax.tick_params(axis='both', which='both', direction='in', labeltop='on')
    norm = mpl.colors.Normalize(vmin=0, vmax=t[-1]*1e9)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,orientation='horizontal')
    cbar.set_label(r"$t\,(10^{-9}\,\mathrm{s})$",labelpad=10)
    cax.minorticks_on()
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.xaxis.set_minor_locator(MultipleLocator(0.1))

    # axis labels
    ax.set_xlabel(r"$k\,(\mathrm{cm}^{-1})$")
    #ax.set_ylabel(r"$|\widetilde{f}|^2\,(\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$|\widetilde{{n}}_{{{}}}|^2\,(\mathrm{{cm}}^{{-6}})$".format(f))
    ax.minorticks_on()
    ax.grid(which='both')
    
    plt.savefig(v+f+"_FFT_power_loglog.pdf", bbox_inches='tight')


#data = h5py.File("reduced_data_fft_power_nov4_test_hdf5_chk.h5","r")
data = h5py.File("reduced_data_fft_power_NSM_sim.h5","r")

for v in variables:
    for f in flavors:
        if v+f+"_FFT" in data:
            print(v+f)
            makeplot(v,f, data)

data.close()
