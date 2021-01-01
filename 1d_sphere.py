# Generate the 3x3 plots of 1d flavor-diagonal values.
# run from the directory you want the pdf to be saved
import cartopy.crs as ccrs
import os
import sys
sys.path.append("/global/homes/s/srichers/Emu/Scripts/visualization")
import amrex_plot_tools as amrex

import copy
import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D

dx = 64./1024.
#ix = 500

basedir="/global/project/projectdirs/m3018/Emu/PAPER/1D/converge_direction/1D_dir128"
directory = basedir+"/plt00300" #[basedir+"/plt00000",basedir+"/plt00300",basedir+"/plt04800"]

ds = yt.load(directory)
print(ds.current_time)
ad = ds.all_data()
f01  = ad['boxlib',"Fx01_Re"]
ix = np.argmin(f01)
print("ix=",ix,np.min(f01))

rkey, ikey = amrex.get_3flavor_particle_keys()
idata, rdata = amrex.read_particle_data(directory, ptype="neutrinos")
zvals = rdata[:,rkey["z"]]
locs = np.where((zvals<(ix+1)*dx) & (zvals>ix*dx))
zvals = zvals[locs]
#fee = np.arctan2(rdata[:,rkey["f01_Im"]][locs],rdata[:,rkey["f01_Re"]][locs])
fee = rdata[:,rkey["f01_Re"]][locs]
pupx = rdata[:,rkey["pupx"]][locs]
pupy = rdata[:,rkey["pupy"]][locs]
pupz = rdata[:,rkey["pupz"]][locs]
pupt = rdata[:,rkey["pupt"]][locs]
xhat = pupx/pupt
yhat = pupy/pupt
zhat = pupz/pupt
latitude = np.arccos(zhat)-np.pi/2.
longitude = np.arctan2(yhat,xhat)
print(np.min(fee),np.max(fee))

# get latitude averages
pzset = set(pupz)
nz = len(pzset)
avgfee = np.zeros(nz)
count = np.zeros(nz)
fee_subtract_average = copy.deepcopy(fee)
for i,pz in zip(range(nz),pzset):
    fee_this_latitude = fee[np.where(pupz==pz)]
    avgfee[i] = np.sum(fee_this_latitude) / len(fee_this_latitude)
    fee_subtract_average[np.where(pupz==pz)] -= avgfee[i]
fee_subtract_average *= 1e6

################
# plot options #
################
mpl.rcParams['font.size'] = 18
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

#Set colours and render
fig, axes = plt.subplots(2,1, figsize=(8,8),subplot_kw={'projection':"mollweide"})
#ax = fig.add_subplot(121,projection="mollweide")

sc0 = axes[0].scatter(longitude, latitude, c=fee, cmap=mpl.cm.plasma, s=3)

# colorbar
cbar = fig.colorbar(sc0,ax=axes[0])
cbar.set_label(r"$\mathrm{Re}(\rho_{e\mu})$")
cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='y', which='both', direction='in')

sc1 = axes[1].scatter(longitude, latitude, c=fee_subtract_average, cmap=mpl.cm.seismic, s=3)#,vmin=-2,vmax=2)

# colorbar
cbar = fig.colorbar(sc1,ax=axes[1])
cbar.set_label(r"$10^6\times\mathrm{Re}\left(\rho_{e\mu}-\langle \rho_{e\mu}\rangle_\phi\right)$")
cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='y', which='both', direction='in')

#ax.set_aspect("equal")
for ax in axes:
    ax.grid(True)
    ax.text(0,0,r"$\hat{x}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    ax.text(.999*np.pi,0,r"$-\hat{x}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    ax.text(-.999*np.pi,0,r"$-\hat{x}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    ax.text(np.pi/2.,0,r"$\hat{y}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    ax.text(-np.pi/2.,0,r"$-\hat{y}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    ax.text(0,np.pi/2.,r"$\hat{z}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    ax.text(0,-np.pi/2.,r"$-\hat{z}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

plt.savefig("1d_sphere_"+directory[-5:]+".pdf",bbox_inches="tight")

