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

cmap = mpl.cm.gist_heat_r

def average_data(filename):
    f = h5py.File(filename,"r")
    phat = np.array(f["phat"])
    Nrho = np.real(np.array(f["Nrho (1|ccm)"]))
    t = np.array(f["t"])
    f.close()
    print("Nrho shape:",np.shape(Nrho))

    # calculate unit momenta
    px = phat[:,0]
    py = phat[:,1]
    pxy = np.minimum(np.sqrt(px**2 + py**2),1.0)
    pz = np.sqrt(1.0 - pxy)
    nparticles = len(pz)
    pz[:nparticles//2] *= -1

    # calculate  longitude and latitude from phat
    latitude = pz #np.arccos(pz) - np.pi/2.
    latitude[np.where(np.abs(latitude)<1e-5)] = 0
    longitude = np.arctan2(py,px)
    unique_lats = np.array(sorted(np.unique(latitude)))
    remove_ids = []
    for i in range(len(unique_lats)):
        locs = np.where(np.isclose(unique_lats[i], unique_lats))[0][1:]
        for loc in locs:
            if(loc not in remove_ids and loc>i):
                remove_ids.append(loc)
    unique_lats = np.delete(unique_lats, remove_ids)

    # integrate the distribution over phi
    shape = np.shape(Nrho)
    integrated_Nrho = np.zeros((shape[0], shape[1], shape[2], len(unique_lats) ))
    for ilat in range(len(unique_lats)):
        locs = np.where(np.isclose(latitude,unique_lats[ilat]))[0]
        integrated_Nrho[:,:,:,ilat] = np.sum(Nrho[:,:,:,locs], axis=3)

    return shape, t, integrated_Nrho, unique_lats

def makeplot():
    
    # set up the figure
    fig, axes = plt.subplots(2,1, figsize=(6,8), sharey=True,sharex=True)
    plt.subplots_adjust(hspace=0,wspace=0)

    # plot the 3D data    
    findices = [0,3,5]

    axes[0].text(0,.5,"crossing",rotation=-90,color="gray")
    axes[0].axhline(1./3.,color="green")
    axes[0].axvline(0,color="gray")
    axes[0].text(.25,.8,"Fiducial")

    ################
    # TWOTHIRDS 1D #
    ################
    shape, t, integrated_Nrho, unique_lats = average_data("/global/project/projectdirs/m3761/PAPER2/TwoThirds_1D/reduced_data_angular_power_spectrum.h5")
    trace = np.sum(integrated_Nrho[0,0,findices,:], axis=0)

    # draw location of crossing
    n_nue = integrated_Nrho[0,0,0,:]-integrated_Nrho[0,1,0,:]
    print(n_nue)
    icross = len(n_nue[np.where(n_nue<0)])
    lat_crossing = 0.5 * (unique_lats[icross]+unique_lats[icross-1])
    axes[1].axhline(1./3.,color="green")
    axes[1].axvline(lat_crossing,color="gray")
    axes[1].text(.25,.8,"TwoThirds")

    axes[1].scatter(unique_lats, integrated_Nrho[-1,0,0,:]/trace, marker="x", color="gray")

    tmax = 5e-9
    
    ################
    # TWOTHIRDS 3D #
    ################
    shape, t, integrated_Nrho, unique_lats = average_data("/global/project/projectdirs/m3761/PAPER2/TwoThirds_3D/1/reduced_data_angular_power_spectrum.h5")
    trace = np.sum(integrated_Nrho[0,0,findices,:], axis=0)
    #for it in range(shape[0]):
    #    toplot = integrated_Nrho[it,0,0,:]
    #    axes[1].plot(unique_lats, toplot/trace, color=cmap(t[it]/tmax))
    toplot = integrated_Nrho[-1,0,0,:]
    axes[1].plot(unique_lats, toplot/trace, color="blue")
    
    ################
    # TWOTHIRDS 2D #
    ################
    shape, t, integrated_Nrho, unique_lats = average_data("/global/project/projectdirs/m3761/PAPER2/TwoThirds_2D/reduced_data_angular_power_spectrum.h5")
    trace = np.sum(integrated_Nrho[0,0,findices,:], axis=0)
    axes[1].scatter(unique_lats, integrated_Nrho[-1,0,0,:]/trace,marker="x", color="black")

    ###############
    # FIDUCIAL 3D #
    ###############
    shape, t, integrated_Nrho, unique_lats = average_data("/global/project/projectdirs/m3761/PAPER2/Fiducial_3D/reduced_data_angular_power_spectrum_withphat.h5")
    trace = np.sum(integrated_Nrho[0,0,findices,:], axis=0)
    #for it in range(shape[0]):
    #    toplot = integrated_Nrho[it,0,0,:]
    #    axes[0].plot(unique_lats, toplot/trace, color=cmap(t[it]/tmax))
    toplot = integrated_Nrho[-1,0,0,:]
    axes[0].plot(unique_lats, toplot/trace, color="blue", label="3D")
    
    ###############
    # FIDUCIAL 2D #
    ###############
    shape, t, integrated_Nrho, unique_lats = average_data("/global/project/projectdirs/m3761/PAPER2/Fiducial_2D/reduced_data_angular_power_spectrum.h5")
    trace = np.sum(integrated_Nrho[0,0,findices,:], axis=0)
    axes[0].scatter(unique_lats, integrated_Nrho[-1,0,0,:]/trace,marker="x", color="black", label="2D")

    ###############
    # FIDUCIAL 1D #
    ###############
    shape, t, integrated_Nrho, unique_lats = average_data("/global/project/projectdirs/m3761/PAPER2/Fiducial_1D/reduced_data_angular_power_spectrum.h5")
    trace = np.sum(integrated_Nrho[0,0,findices,:], axis=0)
    axes[0].scatter(unique_lats, integrated_Nrho[-1,0,0,:]/trace, marker="x", color="gray", label="1D")

    axes[0].legend(frameon=False,fontsize=16,loc="upper left")
                
    # colorbar
    #vmin = 0
    #vmax = tmax * 1e9
    #norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    #cax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
    ##cax.xaxis.set_ticks_position('top')
    ##cax.xaxis.set_label_position('top')
    #cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    #cbar.set_label(r"$t\,\,(\mathrm{ns})$")
    #cbar.ax.minorticks_on()
    #cbar.ax.tick_params(axis='y', which='both', direction='in')
    
    for ax in axes.flatten():
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',right=True,top=True)
        ax.set_xlim(-0.999, 1.0)
        #axes.set_aspect("equal")
        #ax.grid(True)
        ax.set_ylabel(r"$\langle\rho_{ee}\rangle$")
    axes[0].set_xlim(-1,1)

    axes[1].set_xlabel(r"$p_z/|p|$")
    
    
    plt.savefig("averaged_flavor_evolution.pdf",bbox_inches="tight")

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
