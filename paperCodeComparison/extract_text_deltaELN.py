import h5py
import numpy as np

mu_time = 9.4127e10 # 1/s
clight = 2.99792458e10 # cm/s
mu_length = mu_time / clight # 1/cm
print("mu_time = ",mu_time,"1/s")
print("mu_length = ",mu_length,"1/cm")
ndens = 4.89e32

# read data from file
f = h5py.File("reduced_data.h5","r")
t = np.array(f["t"])
N    = np.array(f["N_avg_mag"][:,0,0])
Nbar = np.array(f["Nbar_avg_mag"][:,0,0])
f.close()

# initial number density
# Calculate once rather than at every timestep
# even though it should not change. This way
# errors in the net number density are reflected
# more accurately in the ELN error
Ntot0 = N[0] + Nbar[0]

# calculate the ELN error
ELN = N - Nbar
error = (ELN - ELN[0]) / Ntot0

outfile = open("deltaELN-t-Richers.txt","w")

for i in range(len(ELN)):
    outfile.write(str(t[i]*mu_time)+"\t"+str(error[i])+"\n")

outfile.close()
