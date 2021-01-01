# generate 2x3 convergence plots. Run from PAPER/1D
# run from the script directory

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator, ScalarFormatter)

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
    Nbaravg = np.sum(f["Nbar_avg_mag"][istart:], axis=0) / n
    f.close()

    return Navg, Nbaravg

def get_metrics(dir_list):
    avg_list = []
    avgbar_list = []
    for dirname in dir_list:
        filename = dirname+"/reduced_data.h5"
        Navg,Nbaravg = calc_average_single(filename)
        avg_list.append(Navg)
        avgbar_list.append(Nbaravg)

    return np.array(avg_list), np.array(avgbar_list)


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

ax = plt.gca()
plt.minorticks_on()
basedir="/global/project/projectdirs/m3018/Emu/PAPER/1D"
dir_list = [
    basedir+"/nbar_dens/0.0",
    basedir+"/nbar_dens/0.1",
    basedir+"/nbar_dens/0.2",
    basedir+"/nbar_dens/0.3",
    basedir+"/nbar_dens/0.4",
    basedir+"/nbar_dens/0.5",
    basedir+"/nbar_dens/0.6",
    basedir+"/nbar_dens/0.7",
    basedir+"/nbar_dens/0.8",
    basedir+"/nbar_dens/0.9",
    basedir+"/1D_fiducial"]
x_list = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
avg_list, avgbar_list = get_metrics(dir_list)
plt.plot(x_list, avg_list[:,0,0], color="black", marker="o")
plt.plot(x_list, avgbar_list[:,0,0], color="black", marker="o", linestyle="--")

plt.xlabel(r"$\mathrm{Tr}(\bar{N})/\mathrm{Tr}(N)$")
plt.ylabel(r"$\langle N_{ee}\rangle/\mathrm{Tr}(N)$")

ax.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
ax.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=True)
ax.set_xlim(0,1)
ax.set_ylim(.3,1.05)

plt.savefig("1d_nbardens.pdf", bbox_inches='tight')

