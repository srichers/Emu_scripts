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

filename_list = [(6,"/global/project/projectdirs/m3761/PAPER2/90Degree_1D/reduced_data_angular_power_spectrum.h5"),
                 (6,"/global/project/projectdirs/m3761/PAPER2/90Degree_2D_outplane/reduced_data_angular_power_spectrum.h5"),
                 (6,"/global/project/projectdirs/m3761/PAPER2/90Degree_3D/run1/reduced_data_angular_power_spectrum.h5")]

def ELN(longitude,latitude):
    angle_from_z = np.pi/2. - latitude
    z = np.cos(angle_from_z)
    x = np.sin(angle_from_z) * np.cos(longitude)
    y = np.sin(angle_from_z) * np.sin(longitude)
    ne = 1+z
    nebar = 1+x
    result = ne-nebar
    return result
    
def makeplot():
    fig, axes = plt.subplots(3,1, figsize=(4,8),subplot_kw={'projection':"mollweide"})
    plt.subplots_adjust(hspace=0, wspace=0)
    for iplot in range(3):
        (isnapshot,filename) = filename_list[iplot]
        print(iplot, isnapshot, filename)
        f = h5py.File(filename,"r")
        phat = f["phat"]
        px = phat[:,0]
        py = phat[:,1]
        pz = phat[:,2]
        #pxy = np.minimum(np.sqrt(px**2 + py**2),1.0)
        #pz = np.sqrt(1.0 - pxy)
        #nparticles = len(pz)
        #pz[:nparticles//2] *= -1

        # calculate  longitude and latitude from phat
        latitude = np.pi/2. - np.arccos(pz)
        longitude = np.arctan2(py,px)

        Nrho = f["Nrho (1|ccm)"]

        n_nue      = np.real(Nrho[isnapshot,0,0,:])
        n_numu     = np.real(Nrho[isnapshot,0,3,:])
        n_nutau    = np.real(Nrho[isnapshot,0,5,:])
        n_nuebar   = np.real(Nrho[isnapshot,1,0,:])
        n_numubar  = np.real(Nrho[isnapshot,1,3,:])
        n_nutaubar = np.real(Nrho[isnapshot,1,5,:])
        trace = np.array([n_nue+n_numu+n_nutau,
                          n_nuebar+n_numubar+n_nutaubar])

        # the 0 accesses the ee component
        toplot = np.real(Nrho[isnapshot,0,0,:])/trace[0]
        sc0 = axes[iplot].scatter(longitude, latitude, c=toplot, cmap=cmap, s=5, vmin=vmin, vmax=vmax)
        
        # contour plot
        longitude= np.linspace(-np.pi, np.pi, 40)
        latitude = np.linspace(-np.pi/2, np.pi/2, 20)
        X,Y = np.meshgrid(longitude,latitude)
        Z = ELN(X,Y)
        axes[iplot].contour(X,Y,Z,[0,],colors="k",alpha=0.5,linewidths=2)

        f.close()
        
    # colorbar
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    #cax.xaxis.set_ticks_position('top')
    #cax.xaxis.set_label_position('top')
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, orientation="horizontal")
    cbar.set_label(r"$\langle \rho_{ee} \rangle$")
    cbar.ax.minorticks_on()
    cbar.ax.tick_params(axis='x', which='both', direction='in')

    for ax in axes.flatten():
        #axes.set_aspect("equal")
        ax.grid(True)
        ax.text(0          ,0,r"$\hat{x}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
        ax.text(.999*np.pi ,0,r"$-\hat{x}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
        ax.text(-.999*np.pi,0,r"$-\hat{x}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
        ax.text(np.pi/2.   ,0,r"$\hat{y}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
        ax.text(-np.pi/2.  ,0,r"$-\hat{y}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
        ax.text(0          ,np.pi/2.,r"$\hat{z}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
        ax.text(0          ,-np.pi/2.,r"$-\hat{z}$",horizontalalignment='center',verticalalignment='center',bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    axes[0].text(0,.9,"1D", transform=axes[0].transAxes)
    axes[1].text(0,.9,"2D", transform=axes[1].transAxes)
    axes[2].text(0,.9,"3D", transform=axes[2].transAxes)
        
    plt.savefig("flavor_mollweide.png",bbox_inches="tight")


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
makeplot()
