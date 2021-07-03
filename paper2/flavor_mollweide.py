# Generate the 3x3 plots of 1d flavor-diagonal values.
# run from the directory you want the pdf to be saved
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

cmap = mpl.cm.nipy_spectral_r
vmin = 0
vmax = 1
norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)

def makeplot():
    f = h5py.File("reduced_data_angular_power_spectrum.h5","r")
    phat = f["phat"]
    px = phat[:,0]
    py = phat[:,1]
    pxy = np.minimum(np.sqrt(px**2 + py**2),1.0)
    pz = np.sqrt(1.0 - pxy)
    nparticles = len(pz)
    pz[:nparticles//2] *= -1

    # calculate  longitude and latitude from phat
    latitude = np.arccos(pz) - np.pi/2.
    longitude = np.arctan2(py,px)


    Nrho = f["Nrho (1|ccm)"]

    for isnapshot in range(np.shape(Nrho)[0]):
        plt.close()
        fig, axes = plt.subplots(2,3, figsize=(16,6),subplot_kw={'projection':"mollweide"})

        n_nue      = np.real(Nrho[isnapshot,0,0,:])
        n_numu     = np.real(Nrho[isnapshot,0,3,:])
        n_nutau    = np.real(Nrho[isnapshot,0,5,:])
        n_nuebar   = np.real(Nrho[isnapshot,1,0,:])
        n_numubar  = np.real(Nrho[isnapshot,1,3,:])
        n_nutaubar = np.real(Nrho[isnapshot,1,5,:])
        trace = np.array([n_nue+n_numu+n_nutau,
                          n_nuebar+n_numubar+n_nutaubar])
        print("max nue-nuebar difference:",np.min((n_nue-n_nuebar)/trace[0]),np.min((n_nue-n_nuebar)/trace[1]))

        findices = [0,3,5]
        for im in range(2):
            for fi in range(3):
                toplot = np.real(Nrho[isnapshot,im,findices[fi],:])/trace[im]
                sc0 = axes[im,fi].scatter(longitude, latitude, c=toplot, cmap=cmap, s=5, vmin=vmin, vmax=vmax)

        # colorbar
        cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
        #cax.xaxis.set_ticks_position('top')
        #cax.xaxis.set_label_position('top')
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap)
        cbar.set_label(r"$\langle \rho_{ii} \rangle$")
        cbar.ax.minorticks_on()
        cbar.ax.tick_params(axis='x', which='both', direction='in')

        plt.text(.5,.91,"t = {:.2f}".format(f["t"][isnapshot]*1e9)+" ns",ha='center',va='center',transform=fig.transFigure)
        plt.text(.25,0.05,r"$e$",ha='center',va='center',transform=fig.transFigure)
        plt.text(.5,0.05,r"$\mu$",ha='center',va='center',transform=fig.transFigure)
        plt.text(.78,0.05,r"$\tau$",ha='center',va='center',transform=fig.transFigure)
        plt.text(0.09,0.7,r"$\nu$",ha='center',va='center',transform=fig.transFigure)
        plt.text(0.09,0.3,r"$\bar{\nu}$",ha='center',va='center',transform=fig.transFigure)
        
        for ax in axes.flatten():
            #axes.set_aspect("equal")
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
            
        plt.savefig("flavor_mollweide_"+str(isnapshot).zfill(2)+".png",bbox_inches="tight")
        print("finished", isnapshot)

    f.close()

################
# plot options #
################
mpl.rcParams['font.size'] = 18
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

#Set colours and render
makeplot()
