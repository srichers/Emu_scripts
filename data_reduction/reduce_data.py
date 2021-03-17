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
    ds = yt.load(d)
    t.append(ds.current_time)
    ad = ds.all_data()

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

