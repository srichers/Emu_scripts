import h5py
import numpy as np

mu_time = 9.4127e10 # 1/s
clight = 2.99792458e10 # cm/s
mu_length = mu_time / clight # 1/cm
print("mu_time = ",mu_time,"1/s")
print("mu_length = ",mu_length,"1/cm")
ndens = 4.89e32

f = h5py.File("reduced_data_fft.h5","r")
N = np.array(f["N01_FFT"]) / ndens
phi = np.array(f["N01_FFT_phase"])
k = np.array(f["kz"])/mu_length * 2.*np.pi
f.close()

outfile0 = open("Sfft-k-t0-Richers.txt","w")
outfile1 = open("Sfft-k-t5000-Richers.txt","w")

FFT = N * np.exp(1j * phi) * 2.*np.pi
R = np.real(FFT)
I = np.imag(FFT)

for i in range(len(k)):
    outfile0.write(str(k[i])+"\t"+str(R[ 0,i])+"\t"+str(I[ 0,i])+"\n")
    outfile1.write(str(k[i])+"\t"+str(R[-1,i])+"\t"+str(I[-1,i])+"\n")

outfile0.close()
outfile1.close()
