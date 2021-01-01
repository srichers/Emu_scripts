import numpy as np
import emu_yt_module as emu
import h5py
import glob
import scipy

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

################################
# read data and calculate FFTs #
################################
for d in directories:
    print(d)
    eds = emu.EmuDataset(d)
    t.append(eds.ds.current_time)

    (kx,ky,kz),FFT = eds.fourier("N00_Re")
    N00_FFT.append(FFT)

    (kx,ky,kz),FFT = eds.fourier("N11_Re")
    N11_FFT.append(FFT)

    (kx,ky,kz),FFT = eds.fourier("N22_Re")
    N22_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("N01_Re","N01_Im")
    N01_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("N02_Re","N02_Im")
    N02_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("N12_Re","N12_Im")
    N12_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("Fx00_Re")
    Fx00_FFT.append(FFT)

    (kx,ky,kz),FFT = eds.fourier("Fx11_Re")
    Fx11_FFT.append(FFT)

    (kx,ky,kz),FFT = eds.fourier("Fx22_Re")
    Fx22_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("Fx01_Re","Fx01_Im")
    Fx01_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("Fx02_Re","Fx02_Im")
    Fx02_FFT.append(FFT)
    
    (kx,ky,kz),FFT = eds.fourier("Fx12_Re","Fx12_Im")
    Fx12_FFT.append(FFT)

##################
# write the file #
##################
f = h5py.File("reduced_data_fft.h5","w")
f["t"] = np.array(t)
f["kz"] = np.array(kz)
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
f.close()
