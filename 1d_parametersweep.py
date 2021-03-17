# generate time-series plot
# Run from PAPER/1D/1D_fiducial_long

import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = truncate_colormap(mpl.cm.gist_heat,0,.9)

######################
# read averaged data #
######################
def plotdata(filename):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData["t"])*1e9
    N=np.array(avgData["N_avg_mag"])[:,0,0]
    Nbar=np.array(avgData["Nbar_avg_mag"])[:,0,0]
    avgData.close()
    return t, N, Nbar

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


fig, axes = plt.subplots(4,1, figsize=(6,15))
plt.subplots_adjust(hspace=0,wspace=0)

cbar_height = .17

##############
# plot ndens #
##############
dirlist = [
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/nbar_dens/0.0",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/nbar_dens/0.2",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/nbar_dens/0.4",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/nbar_dens/0.6",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/nbar_dens/0.8",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial"]
value_list = [0,.2,.4,.6,.8,1.0]

for value,dirname in zip(value_list,dirlist):
    t,N,Nbar = plotdata(dirname+"/reduced_data.h5")
    axes[3].plot(t, N,color=cmap(value))
    axes[3].plot(t, Nbar,color=cmap(value), linestyle="--")

ax = fig.add_axes([0,0,0,0])
a = np.array([[0,value_list[-1]]])
img = plt.imshow(a,cmap=cmap, vmax=value_list[-1])
plt.gca().set_visible(False)
cax = fig.add_axes([.95, .12, .06, cbar_height])
cax.tick_params(axis='both', which='both', direction='in')
cbar = plt.colorbar(cax=cax,ticks=value_list)
cbar.set_label(r"$\bar{n}_\mathrm{input}/n_\mathrm{input}$")

################
# plot fluxfac #
################
dirlist = [
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.00",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.05",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.10",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.15",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.20",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.25",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_fluxfac/0.30",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial"]
value_list = np.array([0, .05, .1, .15, .2, .25, .3, 0.33])

for value,dirname in zip(value_list*3,dirlist):
    t,N,Nbar = plotdata(dirname+"/reduced_data.h5")
    axes[1].plot(t, N,color=cmap(value))
    axes[1].plot(t, Nbar,color=cmap(value), linestyle="--")
    
ax = fig.add_axes([0,0,0,1])
a = np.array([[value_list[0],value_list[-1]]])
img = plt.imshow(a,cmap=cmap, vmin=value_list[0],vmax=value_list[-1])
plt.gca().set_visible(False)
cax = fig.add_axes([.95, .51, .06, cbar_height])
cax.tick_params(axis='both', which='both', direction='in')
cbar = plt.colorbar(cax=cax,ticks=value_list)
cbar.set_label(r"$|\bar{\mathbf{f}}_\mathrm{input}|/\bar{n}_\mathrm{input}$")


##################
# plot direction #
##################
dirlist = [
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/0",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/30",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/60",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/90",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/120",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/150",
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial"]
value_list = np.array([0, 30, 60, 90, 120, 150, 180])

for value,dirname in zip(value_list/180.,dirlist):
    t,N,Nbar = plotdata(dirname+"/reduced_data.h5")
    axes[2].plot(t, N,color=cmap(value))
    axes[2].plot(t, Nbar,color=cmap(value), linestyle="--")
    
ax = fig.add_axes([0,0,0,2])
a = np.array([[value_list[0],value_list[-1]]])
img = plt.imshow(a,cmap=cmap, vmin=value_list[0],vmax=value_list[-1])
plt.gca().set_visible(False)
cax = fig.add_axes([.95, .31, .06, cbar_height])
cax.tick_params(axis='both', which='both', direction='in')
cbar = plt.colorbar(cax=cax,ticks=value_list)
cbar.set_label(r"$\cos^{-1}(\hat{\mathbf{f}}_\mathrm{input}\cdot\hat{\bar{\mathbf{f}}}_\mathrm{input})$ (degrees)")


###############
# plot matter #
###############
basedir = "/global/project/projectdirs/m3018/Emu/PAPER/1D/matter"
dirlist = [
    "/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial",
    basedir+"/0.01",
    basedir+"/0.1",
    basedir+"/0.5",
    basedir+"/1",
    basedir+"/2",
    basedir+"/10"]
value_list = np.array([0, .01, 0.1, .5, 1, 2, 10])

vmin = .001
vmax = 10
for value,dirname in zip(value_list,dirlist):
    t,N,Nbar = plotdata(dirname+"/reduced_data.h5")
    cval = np.log10(max(vmin,value)/vmin) / np.log10(vmax/vmin)
    axes[0].plot(t, N,color=cmap(cval))
    axes[0].plot(t, Nbar,color=cmap(cval), linestyle="--")
    
ax = fig.add_axes([0,0,0,3])
a = np.array([[value_list[0],value_list[-1]]])
img = plt.imshow(a,cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
plt.gca().set_visible(False)
cax = fig.add_axes([.95, .7, .06, cbar_height])
cax.tick_params(axis='both', which='both', direction='in')
cbar = plt.colorbar(cax=cax,ticks=value_list)
cbar.set_label(r"$n_e / n_\mathrm{input}$")

##############
# formatting #
##############
for ax in axes:
    ax.axhline(1./3., color="green")
    ax.set_xlabel(r"$t\,(10^{-9}\,\mathrm{s})$")
    ax.set_ylabel(r"$\langle n_{ee}\rangle /\mathrm{Tr}(n_{ab})$")
    ax.set_xlim(.1,5)
    #ax.set_xscale("log")
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

for ax in axes[:3]:
    ax.set_xticklabels([])
    
############
# save pdf #
############
plt.savefig("1d_parametersweep.pdf", bbox_inches="tight")
