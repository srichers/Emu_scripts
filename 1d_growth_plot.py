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

fig, ax = plt.subplots(1,1, figsize=(6,5))

ylabel = r"$\langle |Q|\rangle /\mathrm{Tr}(n_{ab})$"
ax.set_ylabel(ylabel)
ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")    
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.set_yscale("log")
ax.set_ylim(1e-12,.99)
ax.yaxis.set_major_locator(LogLocator(base=10,numticks=20))
ax.yaxis.set_minor_locator(LogLocator(base=10,subs = np.arange(1,10)*.1,numticks=40))

xlim = (0,0.5) # ns

ax.set_xlim(xlim)
ax.plot(t,    N[:,0,1],color="purple",label=r"$n_{e\mu}$")
ax.plot(t,  F[:,0,0,1],color="purple",label=r"$\mathbf{f}^{(x)}_{e\mu}$",linestyle="--")
ax.plot(t,    N[:,0,2],color="green" ,label=r"$n_{e\tau}$")
ax.plot(t,  F[:,0,0,2],color="green" ,label=r"$\mathbf{f}^{(x)}_{e\tau}$",linestyle="--")
ax.plot(t,    N[:,1,2],color="orange",label=r"$n_{\mu\tau}$")
ax.plot(t,  F[:,0,1,2],color="orange",label=r"$\mathbf{f}^{(x)}_{\mu\tau}$",linestyle="--")
ax.legend(frameon=False,ncol=3,fontsize=18,handlelength=.8,handletextpad=0.25,columnspacing=.7)


######################
# theoretical slopes #
######################
w = 6.50e10
t0=.12
t1=.25
y0=5e-5
growth = np.exp(w*(t1-t0)*1e-9)
y1=y0 * growth
ax.plot([t0,t1],[y0,y1],color="blue")
ax.plot([t0,t1],[y0/growth,y1],color="gray")
ax.text(t0-.02,y0*1.5,r"$\mathrm{Im}(\omega)=6.50\times10^{10}\,\mathrm{s}^{-1}$",color="blue",rotation=44,fontsize=12)

w = 1.06e10
t0=.1
t1=.25
y0=8e-8
y1=y0 * np.exp(w*(t1-t0)*1e-9)
#print(w*(t1-t0)*1e-9)
ax.plot([t0,t1],[y0,y1],color="red")
ax.text(t0-.01,y0*1.7,r"$\mathrm{Im}(\omega)=1.06\times10^{10}\,\mathrm{s}^{-1}$",color="red",rotation=8,fontsize=12)

plt.savefig("1d_growth_plot.pdf", bbox_inches="tight")
