import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import sys
sys.path.append("/jet/home/saricher/software/Emu/Scripts/visualization")
import amrex_plot_tools as amrex
import scipy.special
from multiprocessing import Pool
rkey, ikey = amrex.get_3flavor_particle_keys()

# global constants
nproc = 4


# get the max useful number of modes
paramfile = open("inputs","r")
for line in paramfile:
    line_without_comments = line.split("#")[0]
    if "nphi_equator" in line_without_comments:
        nl = int(line_without_comments.split("=")[1]) // 2
paramfile.close()
nl += 1
print("Computing up to l =",nl-1)


############
# GridData #
############
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

#####################################
# spherical_harmonic_power_spectrum #
#####################################
# use scipy.special.sph_harm(m, l, azimuthal_angle, polar_angle)
# np.arctan2(y,x)
def spherical_harmonic_power_spectrum_singlel(l, phi, theta, Nrho):
    nm = 2*l+1
    mlist = np.array(range(nm))-l
    Ylm_star = np.array([np.conj(scipy.special.sph_harm(m, l, phi, theta)) for m in mlist])
    Nrholm_integrand = np.array([Nrho*Ylm_star[im,:] for im in range(nm)])
    Nrholm = np.sum(Nrholm_integrand, axis=3)
    result = np.sum(np.abs(Nrholm)**2, axis=0)
    return result

def spherical_harmonic_power_spectrum(input_data):
    icell = input_data[0]
    idlist = input_data[1]
    p = input_data[2]
    
    # cut out only the bits that we need for this cell
    assert(all(idlist == icell))

    # get direction coordinates
    pupx = p[:,rkey["pupx"]]
    pupy = p[:,rkey["pupy"]]
    pupz = p[:,rkey["pupz"]]
    pupt = p[:,rkey["pupt"]]
    xhat = pupx/pupt
    yhat = pupy/pupt
    zhat = pupz/pupt            
    theta = np.arccos(zhat)
    phi = np.arctan2(yhat,xhat)

    # build Nrho complex values
    nparticles = len(phi)
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
    
    #spectrum = np.zeros((2, 6, nl))
    spectrum = np.array([spherical_harmonic_power_spectrum_singlel(l, phi, theta, Nrho) for l in range(nl)])
    return spectrum


#########################
# loop over directories #
#########################
# get list of directories that contain the neutrino data
directories = sorted(glob.glob("plt*/neutrinos"))
directories = [directories[i].split("/")[0] for i in range(len(directories))]
for directory in directories:
    print()
    print(directory)

    # reset the outputs
    t = []
    spectrum_output = []    

    # don't re-run if data already exists
    outputfilename = "reduced_data_angular_power_spectrum_"+directory[-5:]+".h5"
    if glob.glob(outputfilename)!=[]:
        print(outputfilename+" already exists.")
        print(glob.glob(outputfilename))
        continue
        #quit()

    # get the metadata
    header = amrex.AMReXParticleHeader(directory+"/neutrinos/Header")
    ds = yt.load(directory)
    t.append(ds.current_time)
    ad = ds.all_data()
    grid_data = GridData(ad)
    nlevels = len(header.grids)
    assert nlevels==1
    level = 0
    ngrids = len(header.grids[level])
    
    # average the angular power spectrum over many cells
    # loop over all cells within each grid
    spectrum = np.zeros((nl,2,6))
    total_ncells = 0
    for gridID in range(ngrids):
        print("    grid",gridID+1,"/",ngrids)
        
        # read particle data on a single grid
        idata, rdata = amrex.read_particle_data(directory, ptype="neutrinos", level_gridID=(level,gridID))
        
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
        idlist = [idlist[icell*nppc:(icell+1)*nppc  ] for icell in range(ncells)]
        icell_list = [icell for icell in range(ncells)]
        input_data = zip(range(ncells), idlist, rdata)
        
        # accumulate a spectrum from each cell
        spectrum_each_cell = Pool(nproc).map(spherical_harmonic_power_spectrum, input_data)
        spectrum += np.sum(spectrum_each_cell, axis=0)
        total_ncells += ncells
        
    spectrum /= total_ncells*ad['index',"cell_volume"][0]
    spectrum_output.append(spectrum)

    # write averaged data
    avgData = h5py.File(outputfilename,"w")
    avgData["angular_spectrum"] = spectrum_output
    avgData["t"] = t
    avgData.close()

# import pyshtools as pysh
# https://nbviewer.jupyter.org/github/SHTOOLS/SHTOOLS/blob/master/examples/notebooks/spherical-harmonic-normalizations.ipynb
# lmax = 100
# coeffs = pysh.SHCoeffs.from_zeros(lmax)
# coeffs.set_coeffs(values=[1], ls=[5], ms=[2])
# fig, ax = clm.plot_spectrum(show=False)
# fig, ax = clm.plot_spectrum2d(cmap_rlimits=(1.e-7, 0.1),show=False)



