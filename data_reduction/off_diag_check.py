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
import emu_yt_module_MPC_temp as emu
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

#########################
# average preliminaries #
#########################
def get_matrix_N(suffix):
    assert NF==2
    # need to translate Emu dataset names to FLASH ones
    # WARNING - we are calculating energy densities instead of number densities
    baseFlash = "e"

    if suffix=="":
        suffixFlash = ["e","m","r","p"]
    if suffix=="bar":
        suffixFlash = ["a","n","s","q"]

    #with unit conversions:
    convfact = 4.0*np.pi/e01_energy/(MeV_to_codeenergy/cm_to_codelength**3)#MeV/cm^3/(E code units)

    f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]*convfact
    f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]*convfact
    f01m = ad['flash',baseFlash+suffixFlash[2]+energyGroup]*convfact
    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]*convfact


    f01  =  f01m*np.cos(f01p)
    f01I = -f01m*np.sin(f01p)

    zero = np.zeros(np.shape(f00))

    if(NF==2):
        fR = [[f00 , f01 ], [ f01 ,f11 ]]
        fI = [[zero, f01I], [-f01I,zero]]
    if(NF==3):
        fR = [[f00 , f01 , f02 ], [ f01 ,f11 ,f12 ], [ f02 , f12 ,f22 ]]
        fI = [[zero, f01I, f02I], [-f01I,zero,f12I], [-f02I,-f12I,zero]]

    return fR, fI

def get_matrix_F(base, suffix, NR, NI):
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


    #flux factors:

    f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]
    f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]
    f01m = ad['flash',baseFlash+suffixFlash[2]+energyGroup]
    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]

    #fluences:

    f00 = f00*NR[0][0]
    f11 = f11*NR[1][1]

    Nm = np.sqrt(NR[0][1]**2 + NI[0][1]**2)
    Np = np.arctan2(-NI[0][1], NR[0][1])

    f01  =  f01m*Nm*np.cos(f01p+Np)
    f01I = -f01m*Nm*np.sin(f01p+Np)


    zero = np.zeros(np.shape(f00))

    if(NF==2):
        fR = [[f00 , f01 ], [ f01 ,f11 ]]
        fI = [[zero, f01I], [-f01I,zero]]
    if(NF==3):
        fR = [[f00 , f01 , f02 ], [ f01 ,f11 ,f12 ], [ f02 , f12 ,f22 ]]
        fI = [[zero, f01I, f02I], [-f01I,zero,f12I], [-f02I,-f12I,zero]]

    return fR, fI


def get_mod_F(base, suffix, NR, NI):
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


    #flux factors:

    f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]
    f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]
    f01m = ad['flash',baseFlash+suffixFlash[2]+energyGroup]
    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]

    #fluences:

    #f00 = f00*NR[0][0]
    #f11 = f11*NR[1][1]

    Nm = np.sqrt(NR[0][1]**2 + NI[0][1]**2)
    Np = np.arctan2(-NI[0][1], NR[0][1])

    fmod = f01m*Nm
    #f01  =  f01m*Nm*np.cos(f01p+Np)
    #f01I = -f01m*Nm*np.sin(f01p+Np)


    #zero = np.zeros(np.shape(f00))

    #if(NF==2):
    #    fR = [[f00 , f01 ], [ f01 ,f11 ]]
    #    fI = [[zero, f01I], [-f01I,zero]]
    #if(NF==3):
    #    fR = [[f00 , f01 , f02 ], [ f01 ,f11 ,f12 ], [ f02 , f12 ,f22 ]]
    #    fI = [[zero, f01I, f02I], [-f01I,zero,f12I], [-f02I,-f12I,zero]]

    return fmod

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
                if j != k:
                    Fout[i][j][k] = float(np.sum(np.sqrt( F[i][j][k]**2 + FI[i][j][k]**2))/sumtrace)
                else:
                    Fout[i][j][k] = float(np.sum(F[i][j][k])/sumtrace)

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
    thisN, thisNI = get_matrix_N("")
    #print("N real, min max", np.amin(thisN[0][1]), np.amax(thisN[0][1]))
    #print("N imag, min max", np.amin(thisNI[0][1]), np.amax(thisNI[0][1]))
    print("N mod, min max", np.amin(np.sqrt(thisN[0][1]**2 + thisNI[0][1]**2)), np.amax(np.sqrt(thisN[0][1]**2 + thisNI[0][1]**2)))
    #sumtrace = sumtrace_N(thisN)
    #sumtrace = sumtrace_N(thisN)*NF_2_to_3_nu
    sumtrace = 1.0
    #trace = sumtrace
    N = averaged_N(thisN,thisNI,sumtrace)
    print("N avg mod", N[0][1])

    thisFx, thisFxI = get_matrix_F("Fx","", thisN, thisNI)
    thisFy, thisFyI = get_matrix_F("Fy","", thisN, thisNI)
    thisFz, thisFzI = get_matrix_F("Fz","", thisN, thisNI)
    #print("Fx real, min max", np.amin(thisFx[0][1]), np.amax(thisFx[0][1]))
    #print("Fx imag, min max", np.amin(thisFxI[0][1]), np.amax(thisFxI[0][1]))
    #print("Fy real, min max", np.amin(thisFy[0][1]), np.amax(thisFy[0][1]))
    #print("Fy imag, min max", np.amin(thisFyI[0][1]), np.amax(thisFyI[0][1]))
    #print("Fz real, min max", np.amin(thisFz[0][1]), np.amax(thisFz[0][1]))
    #print("Fz imag, min max", np.amin(thisFzI[0][1]), np.amax(thisFzI[0][1]))
    modFx = get_mod_F("Fx","", thisN, thisNI)
    modFy = get_mod_F("Fy","", thisN, thisNI)
    modFz = get_mod_F("Fz","", thisN, thisNI)
    print("Fx mod, min max", np.amin(modFx), np.amax(modFx))
    print("Fy mod, min max", np.amin(modFy), np.amax(modFy))
    print("Fz mod, min max", np.amin(modFz), np.amax(modFz))
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    F = averaged_F(Ftmp, FtmpI,sumtrace)
    print("Fx avg mod", F[0][0][1])
    print("Fy avg mod", F[1][0][1])
    print("Fz avg mod", F[2][0][1])

    thisN, thisNI = get_matrix_N("bar")
    #print("Nbar real, min max", np.amin(thisN[0][1]), np.amax(thisN[0][1]))
    #print("Nbar imag, min max", np.amin(thisNI[0][1]), np.amax(thisNI[0][1]))
    print("Nbar mod, min max", np.amin(np.sqrt(thisN[0][1]**2 + thisNI[0][1]**2)), np.amax(np.sqrt(thisN[0][1]**2 + thisNI[0][1]**2)))
    #sumtrace = sumtrace_N(thisN)
    ##sumtrace = sumtrace_N(thisN)*NF_2_to_3_bnu
    sumtrace = 1.0
    Nbar = averaged_N(thisN,thisNI,sumtrace)
    print("Nbar avg mod", Nbar[0][1])

    thisFx, thisFxI = get_matrix_F("Fx","bar", thisN, thisNI)
    thisFy, thisFyI = get_matrix_F("Fy","bar", thisN, thisNI)
    thisFz, thisFzI = get_matrix_F("Fz","bar", thisN, thisNI)
    #print("Fxbar real, min max", np.amin(thisFx[0][1]), np.amax(thisFx[0][1]))
    #print("Fxbar imag, min max", np.amin(thisFxI[0][1]), np.amax(thisFxI[0][1]))
    #print("Fybar real, min max", np.amin(thisFy[0][1]), np.amax(thisFy[0][1]))
    #print("Fybar imag, min max", np.amin(thisFyI[0][1]), np.amax(thisFyI[0][1]))
    #print("Fzbar real, min max", np.amin(thisFz[0][1]), np.amax(thisFz[0][1]))
    #print("Fzbar imag, min max", np.amin(thisFzI[0][1]), np.amax(thisFzI[0][1]))
    modFx = get_mod_F("Fx","", thisN, thisNI)
    modFy = get_mod_F("Fy","", thisN, thisNI)
    modFz = get_mod_F("Fz","", thisN, thisNI)
    print("Fx mod, min max", np.amin(modFx), np.amax(modFx))
    print("Fy mod, min max", np.amin(modFy), np.amax(modFy))
    print("Fz mod, min max", np.amin(modFz), np.amax(modFz))
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    Fbar = averaged_F(Ftmp, FtmpI,sumtrace)
    print("Fbarx avg mod", Fbar[0][0][1])
    print("Fbary avg mod", Fbar[1][0][1])
    print("Fbarz avg mod", Fbar[2][0][1])

    print(" ")

