# note - does not work (memory allocation issue)
# have to do it per-

import os
import sys
sys.path.append("/global/homes/s/srichers/emu_scripts/data_reduction")
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
import emu_yt_module as emu

dx = 8./128.
ix=0
iy=0
iz=0
gridID=0

basedir="."
directory = basedir+"/plt01000" #[basedir+"/plt00000",basedir+"/plt00300",basedir+"/plt04800"]

class GridData(object):
    def __init__(self, ad):
        x = ad['index','x'].d
        y = ad['index','y'].d
        z = ad['index','z'].d
        dx = ad['index','dx'].d
        dy = ad['index','dy'].d
        dz = ad['index','dz'].d
        self.ad = ad
        self.dx = dx[0]
        self.dy = dy[0]
        self.dz = dz[0]
        self.xmin = np.min(x-dx/2.)
        self.ymin = np.min(y-dy/2.)
        self.zmin = np.min(z-dz/2.)
        self.xmax = np.max(x+dx/2.)
        self.ymax = np.max(y+dy/2.)
        self.zmax = np.max(z+dz/2.)
        self.nx = int((self.xmax - self.xmin) / self.dx + 0.5)
        self.ny = int((self.ymax - self.ymin) / self.dy + 0.5)
        self.nz = int((self.zmax - self.zmin) / self.dz + 0.5)
        print(self.nx, self.ny, self.nz)
        

    # particle cell id ON THE CURRENT GRID
    # the x, y, and z values are assumed to be relative to the
    # lower boundary of the grid
    def get_particle_cell_ids(self,rdata):
        # get coordinates
        x = rdata[:,rkey["x"]]
        y = rdata[:,rkey["y"]]
        z = rdata[:,rkey["z"]]
        ix = (x/self.dx).astype(int)
        iy = (y/self.dy).astype(int)
        iz = (z/self.dz).astype(int)

        # HACK - get this grid's bounds using particle locations
        ix -= np.min(ix)
        iy -= np.min(iy)
        iz -= np.min(iz)
        nx = np.max(ix)+1
        ny = np.max(iy)+1
        nz = np.max(iz)+1
        idlist = (iz + nz*iy + nz*ny*ix).astype(int)

        return idlist





ds = yt.load(directory)
print(ds.current_time)
ad = ds.all_data()
header = amrex.AMReXParticleHeader(directory+"/neutrinos/Header")
grid_data = GridData(ad)
nlevels = len(header.grids)
assert nlevels==1
level = 0
ngrids = len(header.grids[level])
idata, rdata = amrex.read_particle_data(directory, ptype="neutrinos", level_gridID=(level,gridID))

rkey, ikey = amrex.get_3flavor_particle_keys()
xvals = rdata[:,rkey["x"]]
yvals = rdata[:,rkey["y"]]
zvals = rdata[:,rkey["z"]]
locs = np.where((xvals<(ix+1)*dx) & (xvals>ix*dx) &
                (yvals<(iy+1)*dx) & (yvals>iy*dx) &
                (zvals<(iz+1)*dx) & (zvals>iz*dx) )
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

sc0 = axes[0].scatter(longitude, latitude, c=fee, cmap=mpl.cm.plasma, s=6)

# colorbar
cbar = fig.colorbar(sc0,ax=axes[0])
cbar.set_label(r"$\mathrm{Re}(\rho_{e\mu})$")
cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='y', which='both', direction='in')

sc1 = axes[1].scatter(longitude, latitude, c=fee_subtract_average, cmap=mpl.cm.seismic, s=6)#,vmin=-2,vmax=2)

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

