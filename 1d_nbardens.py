# generate 2x3 convergence plots. Run from PAPER/1D
# run from the script directory

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator, ScalarFormatter)


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

def scale_nubar(n,nbar):
    return (n-nbar)/(nbar)

ax = plt.gca()
basedir="/global/project/projectdirs/m3018/Emu/PAPER/1D"
tstart_nonlinear = 3e-9
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
x_list = np.array([0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
avg_list, avgbar_list = get_metrics(dir_list)
avgbar_list[0] = 1.0
ax.plot(x_list, avg_list[:,0,0], color="black", marker="o", label=r"$n_{ee}$")
n=1.
ax.plot(x_list, avgbar_list[:,0,0]+(1-avg_list[:,0,0])*scale_nubar(n,x_list), color="red", marker="o", linestyle="--", label=r"$\bar{n}_{ee}$")
ax.plot(x_list[-1], avgbar_list[-1,0,0], color="red", marker="o", linestyle="--",markersize=10)
ax.legend(frameon=False, loc="lower left")

#ax.scatter([1.0],avg_list[-1,0,0],marker="*",color="red",s=400)
ax.axhline(1./3., color="green")
ax.text(.78, .34, "Fiducial", color="red", rotation=0, fontsize=16)
ax.text(.38, .6, r"$\mathbf{f}=n/3 \hat{z}$", fontsize=16)
ax.text(.38, .55, r"$\bar{\mathbf{f}}=-\bar{n}/3 \hat{z}$", fontsize=16)

ax.set_xlabel(r"$\bar{n}/n$")
ax.set_ylabel(r"$\langle n_{ee}\rangle/\mathrm{Tr}(n_{ab})$")


ax2 = ax.twiny()
basedir="/global/project/projectdirs/m3018/Emu/PAPER/1D"
tstart_nonlinear = 10e-9
dir_list = [
    basedir+"/rando_test/0.0_thirds",
    basedir+"/rando_test/0.1_thirds",
    basedir+"/rando_test/0.2_thirds",
    basedir+"/rando_test/0.3_thirds",
    basedir+"/rando_test/0.4_thirds",
    basedir+"/rando_test/0.5_thirds",
    basedir+"/rando_test/0.6_thirds",
    basedir+"/rando_test/0.7_thirds",
    basedir+"/rando_test/0.8_thirds",
    basedir+"/rando_test/0.9_thirds",
    basedir+"/rando_test/1.0_thirds"]
x_list = [0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
n=1.
nbar=2./3.
avg_list, avgbar_list = get_metrics(dir_list)
avgbar_list[0] = 1.0
ax2.plot(x_list, avg_list[:,0,0], color="blue", marker="o")
ax2.plot(x_list, avgbar_list[:,0,0]+(1-avg_list[:,0,0])*scale_nubar(n,nbar), color="red", marker="o", linestyle="--")
ax2.set_xticklabels([])
ax2.spines['top'].set_color('blue')
ax2.xaxis.label.set_color('blue')

ax2.set_xlabel(r"$3|\bar{\mathbf{f}}|/\bar{n}$")
ax2.text(.75, .95, r"$\mathbf{f} = 0$", fontsize=16, color="blue")
ax2.text(.75, .9, r"$\hat{\bar{\mathbf{f}}} = -\hat{z}$", fontsize=16, color="blue")
ax2.text(.75, .85, r"$\bar{n}/n = 2/3$", fontsize=16, color="blue")

for axis in [ax,ax2]:
    axis.set_xlim(0,1)
    axis.set_ylim(.3,1.05)
    axis.yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
    axis.minorticks_on()
ax.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=False)
ax2.xaxis.set_tick_params(which='both', direction='in', colors="blue", bottom=False, top=True)


plt.savefig("1d_nbardens.pdf", bbox_inches='tight')

