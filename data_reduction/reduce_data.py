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
def get_matrix(base,suffix):
    assert NF==2
    # need to translate Emu dataset names to FLASH ones
    # WARNING - we are calculating energy densities instead of number densities
    if base=="N":
        baseFlash = "e"
    if base=="Fx":
        baseFlash = "f"
    if base=="Fy":
        baseFlash = "g"
    if base=="Fz":
        baseFlash = "h"
        
    if suffix=="":
        suffixFlash = ["e","m","r","i"]
    if suffix=="bar":
        suffixFlash = ["a","n","s","j"]

    #f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]
    #f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]
    #f01  = ad['flash',baseFlash+suffixFlash[2]+energyGroup]
    #f01I = ad['flash',baseFlash+suffixFlash[3]+energyGroup]

    #with unit conversions:
    if base=="N":
        convfact = 1.0/(MeV_to_codeenergy/cm_to_codelength**3)#MeV/cm^3/(E code units)
    else:
        convfact = 1.0
    f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]*convfact
    f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]*convfact
    f01  = ad['flash',baseFlash+suffixFlash[2]+energyGroup]*convfact
    f01I = ad['flash',baseFlash+suffixFlash[3]+energyGroup]*convfact

    #number densities:
    f00  = 4.0*np.pi*f00/e01_energy
    f11  = 4.0*np.pi*f11/e01_energy
    f01  = 4.0*np.pi*f01/e01_energy
    f01I = 4.0*np.pi*f01I/e01_energy

    zero = np.zeros(np.shape(f00))

    if(NF==2):
        fR = [[f00 , f01 ], [ f01 ,f11 ]]
        fI = [[zero, f01I], [-f01I,zero]]
    if(NF==3):
        fR = [[f00 , f01 , f02 ], [ f01 ,f11 ,f12 ], [ f02 , f12 ,f22 ]]
        fI = [[zero, f01I, f02I], [-f01I,zero,f12I], [-f02I,-f12I,zero]]
    return fR, fI

def sumtrace_N(N):
    sumtrace = 0
    for fi in range(NF):
        sumtrace += np.sum(N[fi][fi])
    return sumtrace

def averaged_N(N, NI, sumtrace):
    R=0
    I=1
    
    # do the averaging
    # f1, f2, R/I
    Nout = np.zeros((NF,NF))
    for i in range(NF):
        for j in range(NF):
            Nout[i][j] = float(np.sum(np.sqrt(N[i][j]**2 + NI[i][j]**2)) / sumtrace)
    return np.array(Nout)

def averaged_F(F, FI, sumtrace):
    R=0
    I=1
    
    # do the averaging
    # direction, f1, f2, R/I
    Fout = np.zeros((3,NF,NF))
    for i in range(3):
        for j in range(NF):
            for k in range(NF):
                Fout[i][j][k] = float(np.sum(np.sqrt( F[i][j][k]**2 + FI[i][j][k]**2))/sumtrace)

    return Fout

def offdiagMag(f):
    R = 0
    I = 1
    result = 0
    for f0 in range(NF):
        for f1 in range(f0+1,NF):
            result += f[:,f0,f1,R]**2 + f[:,f0,f1,I]**2
    return np.sqrt(result)




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
    outputfilename = "reduced_data_"+d
    already_done = len(glob.glob(outputfilename))>0
    if do_average and not already_done:
        thisN, thisNI = get_matrix("N",""   )
        sumtrace = sumtrace_N(thisN)
        #sumtrace = sumtrace_N(thisN)*NF_2_to_3_nu
        trace = sumtrace
        N = averaged_N(thisN,thisNI,sumtrace)

        thisFx, thisFxI = get_matrix("Fx","")
        thisFy, thisFyI = get_matrix("Fy","")
        thisFz, thisFzI = get_matrix("Fz","")
        for f1 in range(2):
            for f2 in range(2):
                thisFx[f1][f2]  = thisFx[f1][f2]  * thisN[f1][f2]
                thisFy[f1][f2]  = thisFy[f1][f2]  * thisN[f1][f2]
                thisFz[f1][f2]  = thisFz[f1][f2]  * thisN[f1][f2]
                thisFxI[f1][f2] = thisFxI[f1][f2] * thisNI[f1][f2]
                thisFyI[f1][f2] = thisFyI[f1][f2] * thisNI[f1][f2]
                thisFzI[f1][f2] = thisFzI[f1][f2] * thisNI[f1][f2]
        Ftmp  = np.array([thisFx , thisFy , thisFz ])
        FtmpI = np.array([thisFxI, thisFyI, thisFzI])
        F = averaged_F(Ftmp, FtmpI,sumtrace)

        thisN, thisNI = get_matrix("N","bar")
        sumtrace = sumtrace_N(thisN)
        #sumtrace = sumtrace_N(thisN)*NF_2_to_3_bnu
        tracebar = sumtrace
        Nbar = averaged_N(thisN,thisNI,sumtrace)

        thisFx, thisFxI = get_matrix("Fx","bar") 
        thisFy, thisFyI = get_matrix("Fy","bar") 
        thisFz, thisFzI = get_matrix("Fz","bar") 
        for f1 in range(2):
            for f2 in range(2):
                thisFx[f1][f2]  = thisFx[f1][f2]  * thisN[f1][f2]
                thisFy[f1][f2]  = thisFy[f1][f2]  * thisN[f1][f2]
                thisFz[f1][f2]  = thisFz[f1][f2]  * thisN[f1][f2]
                thisFxI[f1][f2] = thisFxI[f1][f2] * thisNI[f1][f2]
                thisFyI[f1][f2] = thisFyI[f1][f2] * thisNI[f1][f2]
                thisFzI[f1][f2] = thisFzI[f1][f2] * thisNI[f1][f2]
        Ftmp  = np.array([thisFx , thisFy , thisFz ])
        FtmpI = np.array([thisFxI, thisFyI, thisFzI])
        Fbar = averaged_F(Ftmp, FtmpI,sumtrace)

        print("# rank",mpi_rank,"writing",outputfilename)
        sys.stdout.flush()
        avgData = h5py.File(outputfilename,"w")
        avgData["N_avg_mag"] = [N,]
        avgData["Nbar_avg_mag"] = [Nbar,]
        avgData["F_avg_mag"] = [F,]
        avgData["Fbar_avg_mag"] = [Fbar,]
        avgData["t"] = [t,]
        avgData.close()

    ############
    # FFT work #
    ############
    outputfilename = "reduced_data_fft_power_"+d
    already_done = len(glob.glob(outputfilename))>0
    if do_fft and not already_done:

        print("# rank",mpi_rank,"writing",outputfilename)
        sys.stdout.flush()
        fout = h5py.File(outputfilename,"w")
        fout["t"] = [np.array(t),]

        fft = eds.fourier("ee"+energyGroup,nproc=nproc)
        fout["k"] = get_kmid(fft)
        cleft, cright, ileft, iright, kmid = fft_coefficients(fft)
        N00_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
        fft = eds.fourier("ea"+energyGroup,nproc=nproc)
        N11_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
        fft = eds.fourier("er"+energyGroup,"ei"+energyGroup,nproc=nproc)
        N01_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
        fout["N00_FFT"] = [np.array(N00_FFT),]
        fout["N11_FFT"] = [np.array(N11_FFT),]
        fout["N01_FFT"] = [np.array(N01_FFT),]
        
        fout.close()

