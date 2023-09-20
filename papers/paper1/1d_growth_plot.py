# Create the plot demonstrating exponential growth that matches linear stability analysis
# run from the directory you want the pdf to be saved

import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-uu", "--include_mumu", action="store_true", help="Include mu-mu components.")
parser.add_argument("-tt", "--include_tautau", action="store_true", help="Include tau-tau components.")
parser.add_argument("-bar", "--antineutrinos", action="store_true", help="Plot quantities for antineutrinos instead.")
args = parser.parse_args()

basedir="/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial"


######################
# read averaged data #
######################
avgData = h5py.File(basedir+"/reduced_data.h5","r")

t=np.array(avgData["t"])
locs = t.argsort()
t=t[locs] * 1e9

N=np.array(avgData["N_avg_mag"])[locs]
Nbar=np.array(avgData["Nbar_avg_mag"])[locs]
F=np.array(avgData["F_avg_mag"])[locs]
Fbar=np.array(avgData["Fbar_avg_mag"])[locs]
avgData.close()

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


#############
# MAKE PLOT #
#############

fig, axes = plt.subplots(2,1, figsize=(6,10))
plt.subplots_adjust(hspace=0,wspace=0)

axes[0].set_ylabel(r"$\langle |n_{ab}|\rangle /\mathrm{Tr}(n_{ab})$")
axes[1].set_ylabel(r"$\langle |\mathbf{f}^{(x)}_{ab}|\rangle /\mathrm{Tr}(n_{ab})$")
axes[1].set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
axes[0].set_xticklabels([])

for ax in axes:
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_yscale("log")
    ax.set_ylim(1e-12,.99)
    ax.yaxis.set_major_locator(LogLocator(base=10,numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10,subs = np.arange(1,10)*.1,numticks=40))
    xlim = (0,0.5) # ns
    ax.set_xlim(xlim)

if args.antineutrinos:
    axes[0].plot(t,    Nbar[:,0,1],color="purple",label=r"$\bar{n}_{e\mu}$")
    axes[1].plot(t,  Fbar[:,0,0,1],color="purple",label=r"$\mathbf{\bar{f}}^{(x)}_{e\mu}$")
    axes[0].plot(t,    Nbar[:,0,2],color="green" ,label=r"$\bar{n}_{e\tau}$")
    axes[1].plot(t,  Fbar[:,0,0,2],color="green" ,label=r"$\mathbf{\bar{f}}^{(x)}_{e\tau}$")
    axes[0].plot(t,    Nbar[:,1,2],color="orange",label=r"$\bar{n}_{\mu\tau}$")
    ax.plot(t,  Fbar[:,0,1,2],color="orange",label=r"$\mathbf{\bar{f}}^{(x)}_{\mu\tau}$")
    if args.include_mumu:
        axes[0].plot(t,    Nbar[:,1,1],color="brown",label=r"$\bar{n}_{\mu\mu}$",linestyle="--")
        axes[1].plot(t,  Fbar[:,0,1,1],color="brown",label=r"$\mathbf{\bar{f}}^{(x)}_{\mu\mu}$",linestyle="--")
    if args.include_tautau:
        axes[0].plot(t,    Nbar[:,2,2],color="salmon",label=r"$\bar{n}_{\tau\tau}$",linestyle="--")
        axes[1].plot(t,  Fbar[:,0,2,2],color="salmon",label=r"$\mathbf{\bar{f}}^{(x)}_{\tau\tau}$",linestyle="--")
else:
    axes[0].plot(t,    N[:,0,1],color="purple",label=r"$e\mu$")
    axes[1].plot(t,  F[:,0,0,1],color="purple",label=r"$e\mu$")
    axes[0].plot(t,    N[:,0,2],color="green" ,label=r"$e\tau$")
    axes[1].plot(t,  F[:,0,0,2],color="green" ,label=r"$e\tau$")
    axes[0].plot(t,    N[:,1,2],color="orange",label=r"$\mu\tau$")
    axes[1].plot(t,  F[:,0,1,2],color="orange",label=r"$\mu\tau$")
    if args.include_mumu:
        axes[0].plot(t,    N[:,1,1],color="red",label=r"$\mu\mu$",linestyle="--")
        axes[1].plot(t,  F[:,0,1,1],color="red",label=r"$\mu\mu$",linestyle="--")
    if args.include_tautau:
        axes[0].plot(t,    N[:,2,2],color="gold",label=r"$\tau\tau$",linestyle="--")
        axes[1].plot(t,  F[:,0,2,2],color="gold",label=r"$\tau\tau$",linestyle="--")

if False: #args.include_mumu or args.include_tautau:
    lines = axes[0].get_lines()
    n_lines = []
    f_lines = []

    for i, line in enumerate(lines):
        if i % 2 == 0:
            n_lines.append(line)
        else:
            f_lines.append(line)

    n_legend = plt.legend(n_lines, [l.get_label() for l in n_lines], bbox_to_anchor=(1.0, 0.55), loc="lower right", frameon=False, ncol=2, fontsize=18, handlelength=0.8, handletextpad=0.25, columnspacing=.7)
    #f_legend = plt.legend(f_lines, [l.get_label() for l in f_lines], loc="lower right", frameon=False, ncol=2, fontsize=18, handlelength=0.8, handletextpad=0.25, columnspacing=.7)
    axes[0].add_artist(n_legend)
else:
    axes[0].legend(frameon=False,ncol=2,loc="center right",fontsize=18,handlelength=.8,handletextpad=0.25,columnspacing=.7)



######################
# theoretical slopes #
######################
w = 6.50e10
t0=.12
t1=.25
y0=5e-5
growth = np.exp(w*(t1-t0)*1e-9)
y1=y0 * growth
axes[0].plot([t0,t1],[y0,y1],color="blue")
#ax.plot([t0,t1],[y0/growth,y1],color="gray")
axes[0].text(t0-.02,y0*1.5,r"$\mathrm{Im}(\omega)=6.50\times10^{10}\,\mathrm{s}^{-1}$",color="blue",rotation=44,fontsize=12)

w = 1.06e10
t0=.1
t1=.25
y0=8e-8
y1=y0 * np.exp(w*(t1-t0)*1e-9)
#print(w*(t1-t0)*1e-9)
axes[1].plot([t0,t1],[y0,y1],color="red")
#ax.text(t0-.01,y0*1.7,r"$\mathrm{Im}(\omega)=1.06\times10^{10}\,\mathrm{s}^{-1}$",color="red",rotation=8,fontsize=12)
axes[1].text(t0-.02,y0*1.5,r"$\mathrm{Im}(\omega)=1.06\times10^{10}\,\mathrm{s}^{-1}$",color="red",rotation=8,fontsize=12)

name = "1d_growth_plot"
if args.include_mumu:
    name += "_wmumu"
if args.include_tautau:
    name += "_wtautau"
if args.antineutrinos:
    name += "_bar"
name += ".pdf"

plt.savefig(name, bbox_inches="tight")
