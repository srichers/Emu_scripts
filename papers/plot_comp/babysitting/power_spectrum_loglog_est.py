import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

#variables=["N","Fx","Fy","Fz"]
#flavors=["00","11","22","01","02","12"]
variables=["N"]
flavors=["01"]
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
    it = -1
    total_power = np.sum(fft[it,:])
    ax.loglog(k[:], fft[it,:-1], color='b')
    ind1 = 20
    ind2 = 2
    slope_est = (np.log(fft[it,ind1]) - np.log(fft[it,ind2]))/(np.log(k[ind1]) - np.log(k[ind2]))
    k_line = [k[ind2], k[ind1]]
    fft_line = [10.0*fft[it,ind2], 10.0*fft[it,ind1]]
    ax.loglog(k_line, fft_line, color='orange')

    # axis labels
    ax.set_xlabel(r"$k\,(\mathrm{cm}^{-1})$")
    #ax.set_ylabel(r"$|\widetilde{f}|^2\,(\mathrm{cm}^{-2})$")
    ax.set_ylabel(r"$|\widetilde{{n}}_{{{}}}|^2\,(\mathrm{{cm}}^{{-6}})$".format(f))
    ax.set_title(r"$d\log(|\widetilde{{n}}_{{{}}}|^2)/d\log(k)\simeq{:.2}$".format(f, slope_est))
    ax.minorticks_on()
    ax.grid(which='both')
    
    plt.savefig(v+f+"_FFT_power_loglog_est.pdf", bbox_inches='tight')


data = h5py.File("reduced_data_fft_power_nov4_test_hdf5_chk.h5","r")

for v in variables:
    for f in flavors:
        if v+f+"_FFT" in data:
            print(v+f)
            makeplot(v,f, data)

data.close()
