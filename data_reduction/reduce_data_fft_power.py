import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import emu_yt_module as emu
import h5py
import glob
import scipy
import argparse

k_axis_template = 'z'

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str, default="reduced_data_fft_power.h5", help="Name of the output file (default: reduced_data_fft.h5)")
args = parser.parse_args()

directories = sorted(glob.glob("plt*"))

t = []

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


# use one of the axes as a template for the power spectrum
def ktemplate(fft, axis):
    if(axis=='x'):
        kmid = fft.kx[np.where(fft.kx>=0)]
    if(axis=='y'):
        kmid = fft.ky[np.where(fft.ky>=0)]
    if(axis=='z'):
        kmid = fft.kz[np.where(fft.kz>=0)]
    return kmid

def fft_power(fft,axis):
    kmid = ktemplate(fft,axis)
    
    # add another point to the end of the k grid for interpolation
    # MAKES POWER SPECTRUM HAVE SIZE ONE LARGER THAN KTEMPLATE
    dk = kmid[1]-kmid[0]
    kmid = np.append(kmid, kmid[-1]+dk)
    
    # compute the magnitude of the wavevector for every point
    kmag = 0
    if fft.kx is not None:
        kmag += fft.kx[:,np.newaxis,np.newaxis]**2
    if fft.ky is not None:
        kmag += fft.ky[np.newaxis,:,np.newaxis]**2
    if fft.kz is not None:
        kmag += fft.kz[np.newaxis,np.newaxis,:]**2
    kmag = np.sqrt(kmag)
    kmag[np.where(kmag>kmid[-1])] = kmid[-1]
    
    # compute left index for interpolation
    ileft = (kmag/dk).astype(int)
    iright = ileft+1
    iright[np.where(iright>len(kmid)-1)] = len(kmid)-1

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

################################
# read data and calculate FFTs #
################################
for d in directories:
    print(d)
    eds = emu.EmuDataset(d)
    t.append(eds.ds.current_time)

    fft = eds.fourier("N00_Re")
    N00_FFT.append(fft_power(fft,k_axis_template))    
    fft = eds.fourier("N11_Re")
    N11_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("N22_Re")
    N22_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("N01_Re","N01_Im")
    N01_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("N02_Re","N02_Im")
    N02_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("N12_Re","N12_Im")
    N12_FFT.append(fft_power(fft,k_axis_template))
    
    fft = eds.fourier("Fx00_Re")
    Fx00_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fx11_Re")
    Fx11_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fx22_Re")
    Fx22_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fx01_Re","Fx01_Im")
    Fx01_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fx02_Re","Fx02_Im")
    Fx02_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fx12_Re","Fx12_Im")
    Fx12_FFT.append(fft_power(fft,k_axis_template))
    
    fft = eds.fourier("Fy00_Re")
    Fy00_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fy11_Re")
    Fy11_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fy22_Re")
    Fy22_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fy01_Re","Fy01_Im")
    Fy01_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fy02_Re","Fy02_Im")
    Fy02_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fy12_Re","Fy12_Im")
    Fy12_FFT.append(fft_power(fft,k_axis_template))
    
    fft = eds.fourier("Fz00_Re")
    Fz00_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fz11_Re")
    Fz11_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fz22_Re")
    Fz22_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fz01_Re","Fz01_Im")
    Fz01_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fz02_Re","Fz02_Im")
    Fz02_FFT.append(fft_power(fft,k_axis_template))
    fft = eds.fourier("Fz12_Re","Fz12_Im")
    Fz12_FFT.append(fft_power(fft,k_axis_template))

##################
# write the file #
##################
f = h5py.File(args.output,"w")

f["t"] = np.array(t)
f["k"] = ktemplate(fft,k_axis_template)

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
