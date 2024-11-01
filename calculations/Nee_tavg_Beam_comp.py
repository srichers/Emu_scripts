
import numpy as np
import glob
import h5py


gfermi = 1.1663787e-11 #MeV^{-2}
hbarc = 1.97326966e-11 #MeV cm
hbar = 6.582119569e-22 #MeV s
clight = 29979245800.0 #cm/s

base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]
# real/imag
R=0
I=1
    
t_str = ["t", "t(s)"]
N_str = ["N_avg_mag", "N_avg_mag(1|ccm)"]
#N_str = ["Nbar_avg_mag", "Nbar_avg_mag(1|ccm)"]


######################
# read averaged data #
######################
def read_data(filename,a,b,tind):
    avgData = h5py.File(filename,"r")
    t=np.array(avgData[t_str[tind]])*1e9
    N=np.array(avgData[N_str[tind]])[:,a,b]
    avgData.close()
    return t, N

num_sims = 5
tind = np.ones([2], dtype=np.int64)

emu_base = "/global/cfs/projectdirs/m3761/FLASH/Emu/beam_test/Evan_beam_series_nx1024_Lz8.0/"
emu_h5 = "plt_reduced_data.h5"
emu_sims = ["neebar_{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
emu_inds = -np.ones([num_sims], dtype=np.int32)
emu_inds[0] = 69
emu_inds[1] = 47
emu_inds[2] = 38
emu_inds[3] = 33
emu_inds[4] = 29
tind[0] = 1

#Beam/rand/changing_N_nuebar/*
flash_base = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/MPC/changing_N_nuebar/res_1024/"
flash_h5 = "reduced_data.h5"
flash_sims = ["sim{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
flash_inds = -np.ones([num_sims], dtype=np.int32)
flash_inds[0] = 82
flash_inds[1] = 59
flash_inds[4] = 37
tind[1] = 0

##nuM/Beam/rand/changing_N_nuebar/*
#flash_base = "/global/cfs/projectdirs/m3761/FLASH/FFI_1D/Beam/rand/nuM/changing_N_nuebar/res_1024/"
#flash_h5 = "reduced_data.h5"
#flash_sims = ["sim{:.1f}/".format(0.2*(i+1)) for i in range(num_sims)]
#flash_inds = -np.ones([num_sims], dtype=np.int32)
#flash_inds[1] = 57
#flash_inds[2] = 48
#flash_inds[3] = 40
#tind[1] = 0


for i in range(num_sims):
    print("\\bar{N}_{ee}/N_{ee} = ", 0.2*(i+1))

    filename_emu = emu_base + emu_sims[i] + emu_h5
    t,Nee = read_data(filename_emu,0,0,tind[0])
    t,Nxx = read_data(filename_emu,1,1,tind[0])
    t_ex,N_ex = read_data(filename_emu,0,1,tind[0])
    #normalize emu data:
    Ntr_2f = Nee[0] + Nxx[0]
    Nee = Nee/Ntr_2f
    N_ex = N_ex/Ntr_2f
    if emu_inds[i] != -1:
        tsat_ind = emu_inds[i]
    else:
        tsat_ind = np.argmax(N_ex)
    tstart_ind = 2*tsat_ind
    #trapazoid rule:
    Nee_dt_sum = 0.0
    for j in range(tstart_ind+1,len(Nee)):
        dt = t[j] - t[j-1]
        Nee_dt_sum += 0.5*(Nee[j] + Nee[j-1])*dt
    Nee_avg = Nee_dt_sum/(t[len(Nee)-1] - t[tstart_ind])
    print('Emu, tsat_ind = ', tsat_ind)
    print('Emu, tstart_ind = ', tstart_ind, ", t[tstart_ind] = ", t[tstart_ind], " ns")
    print('Emu, <Nee>/Tr[N] = ', Nee_avg)

    filename_flash = flash_base + flash_sims[i] + flash_h5
    t,Nee = read_data(filename_flash,0,0,tind[1])
    t,N_ex = read_data(filename_flash,0,1,tind[1])
    if flash_inds[i] != -1:
        tsat_ind = flash_inds[i]
    else:
        tsat_ind = np.argmax(N_ex)
    tstart_ind = 2*tsat_ind
    #trapazoid rule:
    Nee_dt_sum = 0.0
    for j in range(tstart_ind+1,len(Nee)):
        dt = t[j] - t[j-1]
        Nee_dt_sum += 0.5*(Nee[j] + Nee[j-1])*dt
    Nee_avg = Nee_dt_sum/(t[len(Nee)-1] - t[tstart_ind])
    print('FLASH, tsat_ind = ', tsat_ind)
    print('FLASH, tstart_ind = ', tstart_ind, ", t[tstart_ind] = ", t[tstart_ind], " ns")
    print('FLASH, <Nee>/Tr[N] = ', Nee_avg, "\n")
