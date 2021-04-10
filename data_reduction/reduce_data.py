# used to make plots but now just generates a hdf5 file with domain-averaged data.
# Run in the directory of the simulation the data should be generated for.
# Still has functionality for per-snapshot plots, but the line is commented out.
# This version averages the magnitudes of off-diagonal components rather than the real/imaginary parts
# also normalizes fluxes by sumtrace of N rather than F.
# This data is used for the growth plot.
# Note - also tried a version using maxima rather than averages, and it did not make the growth plot look any better.

import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import emu_yt_module as emu

# don't re-run if data already exists
outputfilename = "reduced_data.h5"
if glob.glob(outputfilename)!=[]:
    print(outputfilename+" already exists.")
    print(glob.glob(outputfilename))
    quit()


directories = sorted(glob.glob("plt*"))

base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]

N00_FFT = []
N01_FFT = []
N02_FFT = []
N11_FFT = []
N12_FFT = []
N22_FFT = []

Fx00_FFT = []
Fx01_FFT = []
Fx02_FFT = []
Fx11_FFT = []
Fx12_FFT = []
Fx22_FFT = []

Fy00_FFT = []
Fy01_FFT = []
Fy02_FFT = []
Fy11_FFT = []
Fy12_FFT = []
Fy22_FFT = []

Fz00_FFT = []
Fz01_FFT = []
Fz02_FFT = []
Fz11_FFT = []
Fz12_FFT = []
Fz22_FFT = []


def get_kmid(fft):
    if fft.kx is not None:
        kmid = fft.kx[np.where(fft.kx>=0)]
    if fft.ky is not None:
        kmid = fft.ky[np.where(fft.ky>=0)]
    if fft.kz is not None:
        kmid = fft.kz[np.where(fft.kz>=0)]
    return kmid
    
def fft_power(fft):
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

    # compute power contributions to left and right indices
    power = fft.magnitude**2
    powerLeft = power*cleft
    powerRight = power*cright

    # accumulate onto spectrum
    spectrum = np.zeros(len(kmid))
    for i in range(len(kmid)):
        spectrum[i] += np.sum( powerLeft[np.where( ileft==i)])
        spectrum[i] += np.sum(powerRight[np.where(iright==i)])

    return spectrum

def get_matrix(base,suffix):
    f00  = ad['boxlib',base+"00_Re"+suffix]
    f01  = ad['boxlib',base+"01_Re"+suffix]
    f01I = ad['boxlib',base+"01_Im"+suffix]
    f02  = ad['boxlib',base+"02_Re"+suffix]
    f02I = ad['boxlib',base+"02_Im"+suffix]
    f11  = ad['boxlib',base+"11_Re"+suffix]
    f12  = ad['boxlib',base+"12_Re"+suffix]
    f12I = ad['boxlib',base+"12_Im"+suffix]
    f22  = ad['boxlib',base+"22_Re"+suffix]
    zero = np.zeros(np.shape(f00))
    fR = [[f00 , f01 , f02 ], [ f01 ,f11 ,f12 ], [ f02 , f12 ,f22 ]]
    fI = [[zero, f01I, f02I], [-f01I,zero,f12I], [-f02I,-f12I,zero]]
    return fR, fI

def sumtrace_N(N):
    sumtrace = np.sum(N[0][0]+N[1][1]+N[2][2])
    return sumtrace

def averaged_N(N, NI, sumtrace):
    R=0
    I=1
    
    # do the averaging
    # f1, f2, R/I
    Nout = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            Nout[i][j] = float(np.sum(np.sqrt(N[i][j]**2 + NI[i][j]**2)) / sumtrace)
    return np.array(Nout)

def averaged_F(F, FI, sumtrace):
    R=0
    I=1
    
    # do the averaging
    # direction, f1, f2, R/I
    Fout = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Fout[i][j][k] = float(np.sum(np.sqrt( F[i][j][k]**2 + FI[i][j][k]**2))/sumtrace)

    return Fout

def offdiagMag(f):
    R = 0
    I = 1
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)


trace = []
N=[]
F=[]
tracebar = []
Nbar=[]
Fbar=[]
t=[]

for d in directories:
    print(d)
    eds = emu.EmuDataset(d)
    t.append(eds.ds.current_time)
    ad = eds.ds.all_data()

    thisN, thisNI = get_matrix("N",""   )
    sumtrace = sumtrace_N(thisN)
    trace.append(sumtrace)
    N.append(   averaged_N(thisN,thisNI,sumtrace))

    thisN, thisNI = get_matrix("N","bar")
    sumtrace = sumtrace_N(thisN)
    tracebar.append(sumtrace)
    Nbar.append(averaged_N(thisN,thisNI,sumtrace))

    thisFx, thisFxI = get_matrix("Fx","")
    thisFy, thisFyI = get_matrix("Fy","")
    thisFz, thisFzI = get_matrix("Fz","")
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    F.append(averaged_F(Ftmp, FtmpI,sumtrace))
    
    thisFx, thisFxI = get_matrix("Fx","bar") 
    thisFy, thisFyI = get_matrix("Fy","bar") 
    thisFz, thisFzI = get_matrix("Fz","bar") 
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    Fbar.append(averaged_F(Ftmp, FtmpI,sumtrace))

    # FFT stuff
    fft = eds.fourier("N00_Re")
    N00_FFT.append(fft_power(fft))    
    fft = eds.fourier("N11_Re")
    N11_FFT.append(fft_power(fft))
    fft = eds.fourier("N22_Re")
    N22_FFT.append(fft_power(fft))
    fft = eds.fourier("N01_Re","N01_Im")
    N01_FFT.append(fft_power(fft))
    fft = eds.fourier("N02_Re","N02_Im")
    N02_FFT.append(fft_power(fft))
    fft = eds.fourier("N12_Re","N12_Im")
    N12_FFT.append(fft_power(fft))
    
    fft = eds.fourier("Fx00_Re")
    Fx00_FFT.append(fft_power(fft))
    fft = eds.fourier("Fx11_Re")
    Fx11_FFT.append(fft_power(fft))
    fft = eds.fourier("Fx22_Re")
    Fx22_FFT.append(fft_power(fft))
    fft = eds.fourier("Fx01_Re","Fx01_Im")
    Fx01_FFT.append(fft_power(fft))
    fft = eds.fourier("Fx02_Re","Fx02_Im")
    Fx02_FFT.append(fft_power(fft))
    fft = eds.fourier("Fx12_Re","Fx12_Im")
    Fx12_FFT.append(fft_power(fft))
    
    fft = eds.fourier("Fy00_Re")
    Fy00_FFT.append(fft_power(fft))
    fft = eds.fourier("Fy11_Re")
    Fy11_FFT.append(fft_power(fft))
    fft = eds.fourier("Fy22_Re")
    Fy22_FFT.append(fft_power(fft))
    fft = eds.fourier("Fy01_Re","Fy01_Im")
    Fy01_FFT.append(fft_power(fft))
    fft = eds.fourier("Fy02_Re","Fy02_Im")
    Fy02_FFT.append(fft_power(fft))
    fft = eds.fourier("Fy12_Re","Fy12_Im")
    Fy12_FFT.append(fft_power(fft))
    
    fft = eds.fourier("Fz00_Re")
    Fz00_FFT.append(fft_power(fft))
    fft = eds.fourier("Fz11_Re")
    Fz11_FFT.append(fft_power(fft))
    fft = eds.fourier("Fz22_Re")
    Fz22_FFT.append(fft_power(fft))
    fft = eds.fourier("Fz01_Re","Fz01_Im")
    Fz01_FFT.append(fft_power(fft))
    fft = eds.fourier("Fz02_Re","Fz02_Im")
    Fz02_FFT.append(fft_power(fft))
    fft = eds.fourier("Fz12_Re","Fz12_Im")
    Fz12_FFT.append(fft_power(fft))

    kmid = get_kmid(fft)

trace = np.array(trace)
N = np.array(N)
F = np.array(F)
tracebar = np.array(tracebar)
Nbar = np.array(Nbar)
Fbar = np.array(Fbar)

# write averaged data
avgData = h5py.File(outputfilename,"w")
avgData["N_avg_mag"] = N
avgData["Nbar_avg_mag"] = Nbar
avgData["F_avg_mag"] = F
avgData["Fbar_avg_mag"] = Fbar
avgData["t"] = t
avgData.close()

# write FFT file
f = h5py.File("reduced_data_fft_power.h5","w")
f["t"] = np.array(t)
f["k"] = kmid
f["N00_FFT"] = np.array(N00_FFT)
f["N11_FFT"] = np.array(N11_FFT)
f["N22_FFT"] = np.array(N22_FFT)
f["N01_FFT"] = np.array(N01_FFT)
f["N02_FFT"] = np.array(N02_FFT)
f["N12_FFT"] = np.array(N12_FFT)
f["Fx00_FFT"] = np.array(Fx00_FFT)
f["Fx11_FFT"] = np.array(Fx11_FFT)
f["Fx22_FFT"] = np.array(Fx22_FFT)
f["Fx01_FFT"] = np.array(Fx01_FFT)
f["Fx02_FFT"] = np.array(Fx02_FFT)
f["Fx12_FFT"] = np.array(Fx12_FFT)
f["Fy00_FFT"] = np.array(Fy00_FFT)
f["Fy11_FFT"] = np.array(Fy11_FFT)
f["Fy22_FFT"] = np.array(Fy22_FFT)
f["Fy01_FFT"] = np.array(Fy01_FFT)
f["Fy02_FFT"] = np.array(Fy02_FFT)
f["Fy12_FFT"] = np.array(Fy12_FFT)
f["Fz00_FFT"] = np.array(Fz00_FFT)
f["Fz11_FFT"] = np.array(Fz11_FFT)
f["Fz22_FFT"] = np.array(Fz22_FFT)
f["Fz01_FFT"] = np.array(Fz01_FFT)
f["Fz02_FFT"] = np.array(Fz02_FFT)
f["Fz12_FFT"] = np.array(Fz12_FFT)
f.close()
