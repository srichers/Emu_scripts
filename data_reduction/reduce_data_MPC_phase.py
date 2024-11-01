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
import emu_yt_module_MPC as emu
from multiprocessing import Pool
import scipy.special

##########
# INPUTS #
##########
NF = 2
#nproc = 4
nproc = 2
do_average = True
do_fft     = True

#do_MPI = True
do_MPI = False

if(len(sys.argv) != 2):
    print()
    print("Usage: [reduce_data.py filename], where filename is contained in each of the run* subdirectories")
    print()
    exit()

output_base = sys.argv[1]
print(output_base)

energyGroup = "01"

#Scale trace from NF=2 to NF=3 (assumes invariance of trace):
#For test cases, trivial:
#NF_2_to_3_nu = 1.0
#NF_2_to_3_bnu = 1.0
#For NSM:
#NF_2_to_3_nu = 1.2567235951976785
#NF_2_to_3_bnu = 1.204149798567299

#Change yt logging level
yt.set_log_level("error")

#FLASH quantities
e01_energy = emu.e01_energy
MeV_to_codeenergy = emu.nulib_energy_gf
cm_to_codelength = emu.nulib_length_gf

#####################
# FFT preliminaries #
#####################
def get_kmid(fft):
    if fft.kx is not None:
        kmid = fft.kx[np.where(fft.kx>=0)]
    if fft.ky is not None:
        kmid = fft.ky[np.where(fft.ky>=0)]
    if fft.kz is not None:
        kmid = fft.kz[np.where(fft.kz>=0)]
    return kmid

def fft_coefficients(fft):
    # add another point to the end of the k grid for interpolation
    # MAKES POWER SPECTRUM HAVE SIZE ONE LARGER THAN KTEMPLATE
    kmid = get_kmid(fft)
    dk = kmid[1]-kmid[0]
    kmid = np.append(kmid, kmid[-1]+dk)
    
    # compute the magnitude of the wavevector for every point
    kmag = 0
    if fft.kx is not None:
        kmag = kmag + fft.kx[:,np.newaxis,np.newaxis]**2
    if fft.ky is not None:
        kmag = kmag + fft.ky[np.newaxis,:,np.newaxis]**2
    if fft.kz is not None:
        kmag = kmag + fft.kz[np.newaxis,np.newaxis,:]**2
    kmag = np.sqrt(np.squeeze(kmag))
    kmag[np.where(kmag>=kmid[-1])] = kmid[-1]
    
 
    # compute left index for interpolation
    ileft = (kmag/dk).astype(int)
    iright = ileft+1
    iright[np.where(iright>=len(kmid)-1)] = len(kmid)-1

    # compute the fraction of the power that goes toward the left and right k point
    cleft = (kmid[iright]-kmag)/dk
    cright = 1.0-cleft

    return cleft, cright, ileft, iright, kmid

def fft_power(fft, cleft, cright, ileft, iright, kmid):

    # compute power contributions to left and right indices
    power = fft.magnitude**2
    powerLeft = power*cleft
    powerRight = power*cright

    # accumulate onto spectrum
    spectrum = np.array( [ 
        np.sum( powerLeft*(ileft ==i) + powerRight*(iright==i) )
        for i in range(len(kmid))] )

    return spectrum

#########################
# average preliminaries #
#########################
def get_matrix_N_phase(suffix):
    assert NF==2
    # need to translate Emu dataset names to FLASH ones
    # WARNING - we are calculating energy densities instead of number densities
    baseFlash = "e"

    if suffix=="":
        suffixFlash = ["e","m","r","p"]
    if suffix=="bar":
        suffixFlash = ["a","n","s","q"]

    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]


    return f01p

def get_matrix_F_phase(base, suffix):
    assert NF==2
    # need to translate Emu dataset names to FLASH ones
    # WARNING - we are calculating energy densities instead of number densities
    if base=="Fx":
        baseFlash = "f"
    if base=="Fy":
        baseFlash = "g"
    if base=="Fz":
        baseFlash = "h"
        
    if suffix=="":
        suffixFlash = ["e","m","r","p"]
    if suffix=="bar":
        suffixFlash = ["a","n","s","q"]


    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]

    return f01p

def averaged_phase(Min):

    Nout = np.sum(Min.flatten())/float(np.product(np.shape(Min)))

    return np.array(Nout)




#########################
# loop over directories #
#########################
if do_MPI:
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
else:
    mpi_rank = 0
    mpi_size = 1
directories = sorted(glob.glob(output_base+"*"))
#directories = ["nov4_test_hdf5_chk_0375"]
#directories = directories[394:395]
if( (not do_average) and (not do_fft)):
    directories = []
for d in directories[mpi_rank::mpi_size]:
    print("# rank",mpi_rank,"is working on", d)
    sys.stdout.flush()
    eds = emu.EmuDataset(d)
    t = eds.ds.current_time
    ad = eds.ds.all_data()

    ################
    # average work #
    ################
    # write averaged data
    outputfilename = "reduced_data_phase_"+d
    already_done = len(glob.glob(outputfilename))>0
    if do_average and not already_done:
        phaseN = get_matrix_N_phase("")
        Navg = averaged_phase(phaseN)

        orig = get_matrix_F_phase("Fx","") + phaseN
        phaseFx = (orig % (2.0*np.pi)) % (np.pi) \
                - ((orig % (2.0*np.pi))/np.pi).astype(np.int64)*np.pi
        Fxavg = averaged_phase(phaseFx)

        orig = get_matrix_F_phase("Fy","") + phaseN
        phaseFy = (orig % (2.0*np.pi)) % (np.pi) \
                - ((orig % (2.0*np.pi))/np.pi).astype(np.int64)*np.pi
        Fyavg = averaged_phase(phaseFy)

        orig = get_matrix_F_phase("Fz","") + phaseN
        phaseFz = (orig % (2.0*np.pi)) % (np.pi) \
                - ((orig % (2.0*np.pi))/np.pi).astype(np.int64)*np.pi
        Fzavg = averaged_phase(phaseFz)

        Ftmp  = np.array([Fxavg , Fyavg , Fzavg ])


        phaseNbar = get_matrix_N_phase("bar")
        Nbaravg = averaged_phase(phaseNbar)

        orig = get_matrix_F_phase("Fx","bar") + phaseNbar
        phaseFxbar = (orig % (2.0*np.pi)) % (np.pi) \
                - ((orig % (2.0*np.pi))/np.pi).astype(np.int64)*np.pi
        Fxbaravg = averaged_phase(phaseFxbar)

        prig = get_matrix_F_phase("Fy","bar") + phaseNbar
        phaseFybar = (orig % (2.0*np.pi)) % (np.pi) \
                - ((orig % (2.0*np.pi))/np.pi).astype(np.int64)*np.pi
        Fybaravg = averaged_phase(phaseFybar)

        prig = get_matrix_F_phase("Fz","bar") + phaseNbar
        phaseFzbar = (orig % (2.0*np.pi)) % (np.pi) \
                - ((orig % (2.0*np.pi))/np.pi).astype(np.int64)*np.pi
        Fzbaravg = averaged_phase(phaseFzbar)

        Ftmpbar  = np.array([Fxbaravg , Fybaravg , Fzbaravg ])

        print("# rank",mpi_rank,"writing",outputfilename)
        sys.stdout.flush()
        avgData = h5py.File(outputfilename,"w")
        avgData["N_avg_phase"] = [Navg,]
        avgData["Nbar_avg_phase"] = [Nbaravg,]
        avgData["F_avg_phase"] = [Ftmp,]
        avgData["Fbar_avg_phase"] = [Ftmpbar,]
        avgData["t"] = [t,]
        avgData.close()

    #############
    ## FFT work #
    #############
    #outputfilename = "reduced_data_fft_power_"+d
    #already_done = len(glob.glob(outputfilename))>0
    #if do_fft and not already_done:

    #    print("# rank",mpi_rank,"writing",outputfilename)
    #    sys.stdout.flush()
    #    fout = h5py.File(outputfilename,"w")
    #    fout["t"] = [np.array(t),]

    #    fft = eds.fourier("ee"+energyGroup,nproc=nproc)
    #    fout["k"] = get_kmid(fft)
    #    cleft, cright, ileft, iright, kmid = fft_coefficients(fft)
    #    N00_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
    #    fft = eds.fourier("ea"+energyGroup,nproc=nproc)
    #    N11_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
    #    fft = eds.fourier("er"+energyGroup,field_Ph="ep"+energyGroup,nproc=nproc)
    #    N01_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
    #    fout["N00_FFT"] = [np.array(N00_FFT),]
    #    fout["N11_FFT"] = [np.array(N11_FFT),]
    #    fout["N01_FFT"] = [np.array(N01_FFT),]
    #    
    #    fout.close()

