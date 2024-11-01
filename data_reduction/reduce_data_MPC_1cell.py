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

    f00  = np.array(f[baseFlash+suffixFlash[0]+energyGroup])[blk, ind[0], ind[1], ind[2]]*convfact
    f11  = np.array(f[baseFlash+suffixFlash[1]+energyGroup])[blk, ind[0], ind[1], ind[2]]*convfact
    f01m = np.array(f[baseFlash+suffixFlash[2]+energyGroup])[blk, ind[0], ind[1], ind[2]]*convfact
    f01p = np.array(f[baseFlash+suffixFlash[3]+energyGroup])[blk, ind[0], ind[1], ind[2]]


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

    f00  = np.array(f[baseFlash+suffixFlash[0]+energyGroup])[blk, ind[0], ind[1], ind[2]]
    f11  = np.array(f[baseFlash+suffixFlash[1]+energyGroup])[blk, ind[0], ind[1], ind[2]]
    f01m = np.array(f[baseFlash+suffixFlash[2]+energyGroup])[blk, ind[0], ind[1], ind[2]]
    f01p = np.array(f[baseFlash+suffixFlash[3]+energyGroup])[blk, ind[0], ind[1], ind[2]]

    #fluences:

    f00 = f00*NR[0,0]
    f11 = f11*NR[1,1]

    Nm = np.sqrt(NR[0,1]**2 + NI[0,1]**2)
    Np = np.arctan2(-NI[0,1], NR[0,1])

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

def get_zero_matrix(NR):
    assert NF==2

    zero = np.zeros(np.shape(NR[0][0]))

    fR = [[zero, zero], [zero, zero]]
    fI = [[zero, zero], [zero, zero]]

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


def averaged_N_comp(NR, NI, sumtrace):
    
    NRout = np.zeros((NF,NF))
    NIout = np.zeros((NF,NF))
    for i in range(NF):
        for j in range(NF):
            NRout[i][j] = float(NR[i][j]/sumtrace)
            NIout[i][j] = float(NI[i][j]/sumtrace)
    return np.array(NRout), np.array(NIout)

def averaged_N_mag(NR, NI, sumtrace):
    Nout = np.zeros((NF,NF))
    for i in range(NF):
        for j in range(NF):
            Nout[i][j] = float(np.sum(np.sqrt(NR[i][j]**2 + NI[i][j]**2)) / sumtrace)
    return np.array(Nout)

def averaged_F(FR, FI):

    FRout = np.zeros((3,NF,NF))
    FIout = np.zeros((3,NF,NF))
    for i in range(3):
        for j in range(NF):
            for k in range(NF):
                FRout[i][j][k] = float(FR[i][j][k])
                FIout[i][j][k] = float(FI[i][j][k])

    return FRout, FIout


def averaged_F_mag(FR, FI):

    Fout = np.zeros((3,NF,NF))
    for i in range(3):
        for j in range(NF):
            for k in range(NF):
                Fout[i][j][k] = float(np.sum(np.sqrt(FR[i][j][k]**2 + FI[i][j][k]**2)))

    return Fout


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
parser.add_argument('-c', '--cellind', dest='cind', type=str, help='blockID and cell index to use (python counting), e.g., "10 0 4 8"', metavar='', default='random')
parser.add_argument('-d', '--dimensions', dest='dims', type=str, help='Number of dimensions [0-3]', metavar='', default='3')

args = parser.parse_args()

input_base = args.ifb
print('Using ' + input_base + ' for base .hdf5 input files\n')

output_base = args.out
print('Using ' + output_base + ' for base .hdf5 output file\n')

cellstr = args.cind

dimstr = args.dims

filenamelist = sorted(glob.glob(input_base + "*"))

#get number of blocks and size of cell dimensions
f = h5py.File(filenamelist[0],"r")
blocks, nxb, nyb, nzb = get_dom_dims(f["/integer scalars"])
cellsizes = [nxb, nyb, nzb]
f.close()

#parse cellstr to see if it is a legitimate array
cellarr = cellstr.split()

ind = np.zeros([3], dtype=int)

if len(cellarr) == 4:

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

    for j,test in enumerate(cellarr[1:]):

        if RepresentsInt(test):
            cell_ind = int(test)
            if cell_ind < 0:
                cell_ind = cellsizes[j] + cell_ind
            if cell_ind < 0:
                cell_ind = 0
            if cell_ind > (cellsizes[j]-1):
                cell_ind = cellsizes[j] - 1
        else:
            cell_ind = int(np.random.rand()*float(cellsizes[j]-1))
        ind[j] = cell_ind

else:

    blk = int(np.random.rand()*float(blocks-1))

    for j in range(3):
        ind[j] = int(np.random.rand()*float(cellsizes[j]-1))

print('Input block and cell array: ' + cellstr)
print('Using block: {}'.format(blk) + '; cell array: {} {} {}'.format(ind[0], ind[1], ind[2]))


if RepresentsInt(dimstr):
    dims_ind = int(dimstr)
    if dims_ind < 0:
        dims_ind = 3
    if dims_ind > 3:
        dims_ind = 3
else:
    dims_ind = 3
dims = dims_ind

print('Input dimensions: ' + dimstr)
print('Setting number of dimensions: ' + str(dims))

energyGroup = "01"


#FLASH quantities
e01_energy = 50.0 #MeV
nulib_energy_gf = 1.60217733e-6*5.59424238e-55 #code energy/MeV
nulib_length_gf = 6.77140812e-06 #code length/cm

MeV_to_codeenergy = nulib_energy_gf
cm_to_codelength = nulib_length_gf

outputfilename = output_base + '{}_'.format(blk) + 'cell_{}_{}_{}.h5'.format(ind[0], ind[1], ind[2])
out_file = h5py.File(outputfilename,"w")

directories = sorted(glob.glob(input_base+"*"))

tarray = np.empty((len(directories)))

N = np.empty((len(directories),2,2))
Nreal = np.empty((len(directories),2,2))
Nimag = np.empty((len(directories),2,2))
Nbar = np.empty((len(directories),2,2))
Nbarreal = np.empty((len(directories),2,2))
Nbarimag = np.empty((len(directories),2,2))

Flux = np.empty((len(directories),3,2,2))
Freal = np.empty((len(directories),3,2,2))
Fimag = np.empty((len(directories),3,2,2))
Fbar = np.empty((len(directories),3,2,2))
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
    thisN, thisNI = get_matrix_N("")
    sumtrace = sumtrace_N(thisN)
    trace = sumtrace
    NR, NI = averaged_N_comp(thisN,thisNI,sumtrace)
    Nreal[ifile,:,:] = NR
    Nimag[ifile,:,:] = NI
    Nmag = averaged_N_mag(thisN,thisNI,sumtrace)
    N[ifile,:,:] = Nmag

    thisFx, thisFxI = get_zero_matrix(NR)
    thisFy, thisFyI = get_zero_matrix(NR)
    thisFz, thisFzI = get_zero_matrix(NR)
    if dims > 0:
        thisFx, thisFxI = get_matrix_F("Fx","", NR, NI)
        if dims > 1:
            thisFy, thisFyI = get_matrix_F("Fy","", NR, NI)
            if dims > 2:
                thisFz, thisFzI = get_matrix_F("Fz","", NR, NI)
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    FR, FI = averaged_F(Ftmp, FtmpI)
    Fmag = averaged_F_mag(FR, FI)
    Freal[ifile,:,:,:] = FR
    Fimag[ifile,:,:,:] = FI
    Flux[ifile,:,:,:] = Fmag

    thisN, thisNI = get_matrix_N("bar")
    sumtrace = sumtrace_N(thisN)
    tracebar = sumtrace
    NRbar, NIbar = averaged_N_comp(thisN,thisNI,sumtrace)
    Nbarreal[ifile,:,:] = NRbar
    Nbarimag[ifile,:,:] = NIbar
    Nbarmag = averaged_N_mag(thisN,thisNI,sumtrace)
    Nbar[ifile,:,:] = Nbarmag

    thisFx, thisFxI = get_zero_matrix(NRbar)
    thisFy, thisFyI = get_zero_matrix(NRbar)
    thisFz, thisFzI = get_zero_matrix(NRbar)
    if dims > 0:
        thisFx, thisFxI = get_matrix_F("Fx","bar", NRbar, NIbar)
        if dims > 1:
            thisFy, thisFyI = get_matrix_F("Fy","bar", NRbar, NIbar)
            if dims > 2:
                thisFz, thisFzI = get_matrix_F("Fz","bar", NRbar, NIbar)
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    FRbar, FIbar = averaged_F(Ftmp, FtmpI)
    Fbarmag = averaged_F_mag(FRbar, FIbar)
    Fbarreal[ifile,:,:,:] = FRbar
    Fbarimag[ifile,:,:,:] = FIbar
    Fbar[ifile,:,:,:] = Fbarmag

    sys.stdout.flush()


out_file["N_real"] = Nreal
out_file["N_imag"] = Nimag

out_file["N_avg_mag"] = N

out_file["Nbar_real"] = Nbarreal
out_file["Nbar_imag"] = Nbarimag

out_file["Nbar_avg_mag"] = Nbar

out_file["F_real"] = Freal
out_file["F_imag"] = Fimag

out_file["F_avg_mag"] = Flux

out_file["Fbar_real"] = Fbarreal
out_file["Fbar_imag"] = Fbarimag

out_file["Fbar_avg_mag"] = Fbar

out_file["t"] = tarray

out_file.close()


