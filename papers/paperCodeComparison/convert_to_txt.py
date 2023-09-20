import numpy as np
import h5py
import sys

hbar = 1.05457266e-27 # erg s
c = 2.99792458e10 # cm/s
eV = 1.60218e-12 # erg
GeV = 1e9 * eV # erg
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3

ndens = 4.89e32 # cm^-3
mu = np.sqrt(2)*GF*ndens # erg
mu_t = mu / hbar
mu_l = mu_t / c

f = h5py.File(sys.argv[1],"r")
N = np.array(f["N_avg_mag"])
Nbar = np.array(f["Nbar_avg_mag"])
t = np.array(f["t"]) * mu_t
f.close()

fout = open("Psur-t-Richers.txt","w")
for i in range(len(t)):
    fout.write(str(t[i]) + "\t" + str(N[i,0,0]) + "\t" + str(Nbar[i,0,0]) + "\n")
fout.close()

