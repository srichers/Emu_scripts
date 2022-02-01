import h5py
import numpy as np
import matplotlib.pyplot as plt

infile = h5py.File("reduced_data_fft.h5","r")
N01fft = np.array(infile["N01_FFT"])
k = np.array(infile["kz"])
print(np.shape(N01fft))
infile.close()

plt.semilogy(k,N01fft[0])
plt.savefig("spectrum_initial.pdf",bbox_inches="tight")
