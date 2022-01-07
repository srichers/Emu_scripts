# used to make plots but now just generates a hdf5 file with domain-averaged data.
# Run in the directory of the simulation the data should be generated for.
# Still has functionality for per-snapshot plots, but the line is commented out.
# This version averages the magnitudes of off-diagonal components rather than the real/imaginary parts
# also normalizes fluxes by sumtrace of N rather than F.
# This data is used for the growth plot.
# Note - also tried a version using maxima rather than averages, and it did not make the growth plot look any better.

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import amrex_plot_tools as amrex
import emu_yt_module as emu
from multiprocessing import Pool
import scipy.special

##########
# INPUTS #
##########
nproc = 1

#########################
# angular preliminaries #
#########################
rkey, ikey = amrex.get_particle_keys()
paramfile = open("inputs","r")
for line in paramfile:
    line_without_comments = line.split("#")[0]
    if "nphi_equator" in line_without_comments:
        nl = int(line_without_comments.split("=")[1]) // 2
paramfile.close()
nl += 1

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

# get the number of particles per cell
def get_nppc(d):
    eds = emu.EmuDataset(d)
    t = eds.ds.current_time
    ad = eds.ds.all_data()
    grid_data = GridData(ad)
    level = 0
    gridID = 0
    idata, rdata = amrex.read_particle_data(d, ptype="neutrinos", level_gridID=(level,gridID))
    idlist = grid_data.get_particle_cell_ids(rdata)
    ncells = np.max(idlist)+1
    nppc = len(idlist) // ncells
    return nppc

    
# input list of particle data separated into grid cells
# output the same array, but sorted by zenith angle, then azimuthal angle
# also output the grid of directions in each cell (assumed to be the same)
def sort_rdata_chunk(p):
    # sort first in theta
    sorted_indices = p[:,rkey["pupz"]].argsort()
    p = p[sorted_indices,:]

    # loop over unique values of theta
    costheta = p[:,rkey["pupz"]] / p[:,rkey["pupt"]]
    for unique_costheta in np.unique(costheta):
        # get the array of particles with the same costheta
        costheta_locs = np.where(costheta == unique_costheta)[0]
        p_theta = p[costheta_locs,:]
        
        # sort these particles by the azimuthal angle
        phi = np.arctan2(p_theta[:,rkey["pupy"]] , p_theta[:,rkey["pupx"]] )
        sorted_indices = phi.argsort()
        p_theta = p_theta[sorted_indices,:]
        
        # put the sorted data back into p
        p[costheta_locs,:] = p_theta
        
    # return the sorted array
    return p

def get_Nrho(p):
    # build Nrho complex values
    nparticles = len(p)
    if NF==2:
        Nrho = np.zeros((2,3,nparticles))*1j
        Nrho[0,0,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f00_Re"   ]] + 1j*0                      )
        Nrho[0,1,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f01_Re"   ]] + 1j*p[:,rkey["f01_Im"   ]] )
        Nrho[0,2,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f11_Re"   ]] + 1j*0                      )
        Nrho[1,0,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f00_Rebar"]] + 1j*0                      )
        Nrho[1,1,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f01_Rebar"]] + 1j*p[:,rkey["f01_Imbar"]] )
        Nrho[1,2,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f11_Rebar"]] + 1j*0                      )
    if NF==3:
        Nrho = np.zeros((2,6,nparticles))*1j
        Nrho[0,0,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f00_Re"   ]] + 1j*0                      )
        Nrho[0,1,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f01_Re"   ]] + 1j*p[:,rkey["f01_Im"   ]] )
        Nrho[0,2,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f02_Re"   ]] + 1j*p[:,rkey["f02_Im"   ]] )
        Nrho[0,3,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f11_Re"   ]] + 1j*0                      )
        Nrho[0,4,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f12_Re"   ]] + 1j*p[:,rkey["f12_Im"   ]] )
        Nrho[0,5,:] = p[:,rkey["N"   ]] * ( p[:,rkey["f22_Re"   ]] + 1j*0                      )
        Nrho[1,0,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f00_Rebar"]] + 1j*0                      )
        Nrho[1,1,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f01_Rebar"]] + 1j*p[:,rkey["f01_Imbar"]] )
        Nrho[1,2,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f02_Rebar"]] + 1j*p[:,rkey["f02_Imbar"]] )
        Nrho[1,3,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f11_Rebar"]] + 1j*0                      )
        Nrho[1,4,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f12_Rebar"]] + 1j*p[:,rkey["f12_Imbar"]] )
        Nrho[1,5,:] = p[:,rkey["Nbar"]] * ( p[:,rkey["f22_Rebar"]] + 1j*0                      )
    return Nrho

def get_maxDeltaLength(p):
    assert(NF==2)
    rho_z = 0.5 * np.array([ p[:,rkey["f00_Re"]]-p[:,rkey["f11_Re"]], p[:,rkey["f00_Re"]]-p[:,rkey["f11_Re"]] ])
    rho_x = np.array([ p[:,rkey["f01_Re"]], p[:,rkey["f01_Rebar"]] ])
    rho_y = np.array([ p[:,rkey["f01_Im"]], p[:,rkey["f01_Imbar"]] ])
    length = np.sqrt(rho_x**2 + rho_y**2 + rho_z**2)
    deltaLength = np.abs(0.5 - length)
    return np.max(deltaLength)

# get NF
directories = sorted(glob.glob("plt*"))
eds = emu.EmuDataset(directories[0])
NF = eds.get_num_flavors()

# separate loop for angular spectra so there is no aliasing and better load balancing
directories = sorted(glob.glob("plt*/neutrinos"))
directories = [directories[i].split('/')[0] for i in range(len(directories))] # remove "neutrinos"

# get number of particles to be able to construct 
nppc = get_nppc(directories[-1])

# open delta length file
maxDeltaLength_file = open("deltaP-t-Richers.txt","w")

if __name__ == '__main__':
    pool = Pool(nproc)
    for d in directories:
        print("# working on", d)
        eds = emu.EmuDataset(d)
        t = eds.ds.current_time
        ad = eds.ds.all_data()
        dir_number = d[3:]

        ################
        # angular work #
        ################
        header = amrex.AMReXParticleHeader(d+"/neutrinos/Header")
        grid_data = GridData(ad)
        nlevels = len(header.grids)
        assert nlevels==1
        level = 0
        ngrids = len(header.grids[level])
        
        # average the angular power spectrum over many cells
        # loop over all cells within each grid
        if NF==2:
            ncomps = 3
        if NF==3:
            ncomps = 6
        Nrho_avg = np.zeros((2,ncomps,nppc))*1j
        total_ncells = 0
        maxDeltaLength = 0
        for gridID in range(ngrids):
            print("    ","grid",gridID+1,"/",ngrids)
            
            # read particle data on a single grid
            idata, rdata = amrex.read_particle_data(d, ptype="neutrinos", level_gridID=(level,gridID))
            
            # get list of cell ids
            idlist = grid_data.get_particle_cell_ids(rdata)
            
            # sort rdata based on id list
            sorted_indices = idlist.argsort()
            rdata = rdata[sorted_indices]
            idlist = idlist[sorted_indices]
            
            # split up the data into cell chunks
            ncells = np.max(idlist)+1
            nppc = len(idlist) // ncells
            rdata  = [ rdata[icell*nppc:(icell+1)*nppc,:] for icell in range(ncells)]
            chunksize = ncells//nproc
            if ncells % nproc != 0:
                chunksize += 1
                
            # sort particles in each chunk
            rdata = pool.map(sort_rdata_chunk, rdata, chunksize=chunksize)
            phat = rdata[0][:,rkey["pupx"]:rkey["pupz"]+1] / rdata[0][:,rkey["pupt"]][:,np.newaxis]

            # accumulate the spatial average of the angular distribution
            Nrho = pool.map(get_Nrho,rdata, chunksize=chunksize) #
            Nrho_avg += np.sum(Nrho, axis=0)
            
            # count the total number of cells
            total_ncells += ncells

            # write max delta length
            maxDeltaLength = max(maxDeltaLength, np.max(pool.map( get_maxDeltaLength, rdata, chunksize=chunksize)) )

        # write max delta length
        maxDeltaLength_file.write(str(t.d)+"\t"+str(maxDeltaLength)+"\n")
            
        # divide out cell volume and number of cells to get average differential number density
        Nrho_avg /= total_ncells*ad['index',"cell_volume"][0]

        # average azimuthally
        pz = phat[:,2]
        unique_pz = np.unique(pz)
        PolarizationVector    = [np.real(Nrho_avg[0,1,:]), np.imag(Nrho_avg[0,1,:]), 0.5*(Nrho_avg[0,0,:]-Nrho_avg[0,2,:])]
        Pz0 = 0.5 * np.real(Nrho_avg[0,0,:] + Nrho_avg[0,2,:])
        PolarizationVector = PolarizationVector / Pz0
        def average_at_given_pz(pz_target):
            locs = np.where(pz==pz_target)
            npoints = len(locs[0])
            P = np.array([ np.sum(PolarizationVector[i][locs]) for i in range(3) ])
            P = P / npoints
            return P
        P = np.array([average_at_given_pz(pz_target) for pz_target in unique_pz])
        #for i in range(len(unique_pz)):
        #    print(i, unique_pz[i], P[i,:], np.sum(P[i,:]**2))

        
        outputfilename = "P-u-t"+dir_number+".txt"
        print("# writing",outputfilename)
        P_u_file = open(outputfilename,"w")
        for i in range(len(unique_pz)):
            P_u_file.write(str(unique_pz[i])+"\t"+str(P[i,0])+"\t"+str(P[i,1])+"\t"+str(P[i,2])+"\n")
        P_u_file.close()
    

maxDeltaLength_file.close()
