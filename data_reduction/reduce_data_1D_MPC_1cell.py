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
import glob
import multiprocessing as mp
import h5py
import argparse


def get_time(dset):
    for thing in dset:
        if thing[0].strip()==b'time':
            return thing[1]


def get_dom_dims(dset):
    for thing in dset:
        if thing[0].strip()==b'nxb':
            nxb = thing[1]
        elif thing[0].strip()==b'nyb':
            nyb = thing[1]
        elif thing[0].strip()==b'nzb':
            nzb = thing[1]
        elif thing[0].strip()==b'globalnumblocks':
            blocks = thing[1]
    return blocks, nxb, nyb, nzb


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

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
        suffixFlash = ["e","m","r","p"]
    if suffix=="bar":
        suffixFlash = ["a","n","s","q"]

    #with unit conversions:
    if base=="N":
        convfact = 4.0*np.pi/e01_energy/(MeV_to_codeenergy/cm_to_codelength**3)#MeV/cm^3/(E code units)
    else:
        convfact = 1.0
    f00  = np.array(f[baseFlash+suffixFlash[0]+energyGroup])[blk, 0, 0, ind]*convfact
    f11  = np.array(f[baseFlash+suffixFlash[1]+energyGroup])[blk, 0, 0, ind]*convfact
    f01m = np.array(f[baseFlash+suffixFlash[2]+energyGroup])[blk, 0, 0, ind]*convfact
    f01p = np.array(f[baseFlash+suffixFlash[3]+energyGroup])[blk, 0, 0, ind]*convfact


    f01  = f01m*np.cos(f01p)
    f01I = f01m*np.sin(f01p)

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

def offdiagMag(flux):
    R = 0
    I = 1
    result = 0
    for f0 in range(NF):
        for f1 in range(f0+1,NF):
            result += flux[:,f0,f1,R]**2 + flux[:,f0,f1,I]**2
    return np.sqrt(result)


def averaged_N(NR, NI, sumtrace):
    
    NRout = np.zeros((NF,NF))
    NIout = np.zeros((NF,NF))
    for i in range(NF):
        for j in range(NF):
            NRout[i][j] = float(NR[i][j]/sumtrace)
            NIout[i][j] = float(NI[i][j]/sumtrace)
    return np.array(NRout), np.array(NIout)

def averaged_F(FR, FI, sumtrace):

    FRout = np.zeros((1,NF,NF))
    FIout = np.zeros((1,NF,NF))
    for i in range(1):
        for j in range(NF):
            for k in range(NF):
                FRout[i][j][k] = float(FR[i][j][k]/sumtrace)
                FIout[i][j][k] = float(FI[i][j][k]/sumtrace)

    return FRout, FIout



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

parser = argparse.ArgumentParser(description='Reads results_vs_space_time_*.dat file and creates plot')
parser.add_argument('-i', '--infilebase', dest='ifb', type=str, help='base input hdf5 file', metavar='', default='sim_hdf5_chk_')
parser.add_argument('-o', '--outfilebase', dest='out', type=str, help='base output .txt file name', metavar='', default='reduced_data_')
parser.add_argument('-c', '--cellind', dest='cind', type=str, help='blockID and cell index to use (python counting), e.g., "0 1000"', metavar='', default='random')

args = parser.parse_args()

input_base = args.ifb
print('Using ' + input_base + ' for base .hdf5 input files\n')

output_base = args.out
print('Using ' + output_base + ' for base .hdf5 output file\n')

cellstr = args.cind

filenamelist = sorted(glob.glob(input_base + "*"))

#get number of blocks and size of cell dimensions
f = h5py.File(filenamelist[0],"r")
blocks, nxb, nyb, nzb = get_dom_dims(f["/integer scalars"])
cellsize = nxb
f.close()

#parse cellstr to see if it is a legitimate array
cellarr = cellstr.split()

ind = 0

if len(cellarr) == 2:

    if RepresentsInt(cellarr[0]):
        blk_ind = int(cellarr[0])
        if blk_ind < 0:
            blk_ind = blocks + blk_ind
        if blk_ind < 0:
            blk_ind = 0
        if blk_ind > (blocks-1):
            blk_ind = blocks - 1
    else:
        blk_ind = int(np.random.rand()*float(blocks-1))
    blk = blk_ind

    if RepresentsInt(cellarr[1]):
        cell_ind = int(cellarr[1])
        if cell_ind < 0:
            cell_ind = cellsize + cell_ind
        if cell_ind < 0:
            cell_ind = 0
        if cell_ind > (cellsize-1):
            cell_ind = cellsize - 1
    else:
        cell_ind = int(np.random.rand()*float(cellsize-1))
    ind = cell_ind

else:

    blk = int(np.random.rand()*float(blocks-1))

    ind = int(np.random.rand()*float(cellsize-1))

print('Input block and cell array: ' + cellstr)
print('Using block: {}'.format(blk) + '; cell: {}'.format(ind))


energyGroup = "01"


#FLASH quantities
e01_energy = 50.0 #MeV
nulib_energy_gf = 1.60217733e-6*5.59424238e-55 #code energy/MeV
nulib_length_gf = 6.77140812e-06 #code length/cm

MeV_to_codeenergy = nulib_energy_gf
cm_to_codelength = nulib_length_gf

outputfilename = output_base + '{}_'.format(blk) + 'cell_{}.h5'.format(ind)
out_file = h5py.File(outputfilename,"w")

directories = sorted(glob.glob(input_base+"*"))

tarray = np.empty((len(directories)))

Nreal = np.empty((len(directories),2,2))
Nimag = np.empty((len(directories),2,2))
Nbarreal = np.empty((len(directories),2,2))
Nbarimag = np.empty((len(directories),2,2))

Freal = np.empty((len(directories),3,2,2))
Fimag = np.empty((len(directories),3,2,2))
Fbarreal = np.empty((len(directories),3,2,2))
Fbarimag = np.empty((len(directories),3,2,2))

#########################
# loop over directories #
#########################
for ifile, filename in enumerate(directories):
    sys.stdout.flush()

    f = h5py.File(filename,"r")
    tarray[ifile] = get_time(f["/real scalars"])


    # write single cell specific data
    thisN, thisNI = get_matrix("N",""   )
    sumtrace = sumtrace_N(thisN)
    trace = sumtrace
    NR, NI = averaged_N(thisN,thisNI,sumtrace)
    Nreal[ifile,:,:] = NR
    Nimag[ifile,:,:] = NI

    thisFx, thisFxI = get_matrix("Fx","")
    for f1 in range(2):
        for f2 in range(2):
            thisFx[f1][f2]  = thisFx[f1][f2]  * thisN[f1][f2]
            thisFxI[f1][f2] = thisFxI[f1][f2] * thisNI[f1][f2]
    Ftmp  = np.array([thisFx])
    FtmpI = np.array([thisFxI])
    FR, FI = averaged_F(Ftmp, FtmpI,sumtrace)
    Freal[ifile,:,:,:] = FR
    Fimag[ifile,:,:,:] = FI

    thisN, thisNI = get_matrix("N","bar")
    sumtrace = sumtrace_N(thisN)
    #sumtrace = sumtrace_N(thisN)*NF_2_to_3_bnu
    tracebar = sumtrace
    NRbar, NIbar = averaged_N(thisN,thisNI,sumtrace)
    Nbarreal[ifile,:,:] = NRbar
    Nbarimag[ifile,:,:] = NIbar

    thisFx, thisFxI = get_matrix("Fx","bar") 
    for f1 in range(2):
        for f2 in range(2):
            thisFx[f1][f2]  = thisFx[f1][f2]  * thisN[f1][f2]
            thisFxI[f1][f2] = thisFxI[f1][f2] * thisNI[f1][f2]
    Ftmp  = np.array([thisFx])
    FtmpI = np.array([thisFxI])
    FRbar, FIbar = averaged_F(Ftmp, FtmpI,sumtrace)
    Fbarreal[ifile,:,:,:] = FRbar
    Fbarimag[ifile,:,:,:] = FIbar

    sys.stdout.flush()


out_file["N_real"] = Nreal
out_file["N_imag"] = Nimag

out_file["Nbar_real"] = Nbarreal
out_file["Nbar_imag"] = Nbarimag

out_file["F_real"] = Freal
out_file["F_imag"] = Fimag

out_file["Fbar_real"] = Fbarreal
out_file["Fbar_imag"] = Fbarimag

out_file["t"] = tarray

out_file.close()


