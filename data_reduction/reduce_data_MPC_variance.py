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
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import amrex_plot_tools as amrex
import emu_yt_module_MPC as emu
from multiprocessing import Pool
import scipy.special

##########
# INPUTS #
##########
NF = 2
#nproc = 4
nproc = 2
do_average = True
#do_fft     = True
do_fft     = False

#do_MPI = True
do_MPI = False

if(len(sys.argv) != 2):
    print()
    print("Usage: [reduce_data.py filename], where filename is contained in each of the run* subdirectories")
    print()
    exit()

output_base = sys.argv[1]
print(output_base)

energyGroup = "01"

#Scale trace from NF=2 to NF=3 (assumes invariance of trace):
#For test cases, trivial:
#NF_2_to_3_nu = 1.0
#NF_2_to_3_bnu = 1.0
#For NSM:
#NF_2_to_3_nu = 1.2567235951976785
#NF_2_to_3_bnu = 1.204149798567299

#Change yt logging level
yt.set_log_level("error")

#FLASH quantities
e01_energy = emu.e01_energy
MeV_to_codeenergy = emu.nulib_energy_gf
cm_to_codelength = emu.nulib_length_gf

#####################
# FFT preliminaries #
#####################
def get_kmid(fft):
    if fft.kx is not None:
        kmid = fft.kx[np.where(fft.kx>=0)]
    if fft.ky is not None:
        kmid = fft.ky[np.where(fft.ky>=0)]
    if fft.kz is not None:
        kmid = fft.kz[np.where(fft.kz>=0)]
    return kmid

def fft_coefficients(fft):
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

    return cleft, cright, ileft, iright, kmid

def fft_power(fft, cleft, cright, ileft, iright, kmid):

    # compute power contributions to left and right indices
    power = fft.magnitude**2
    powerLeft = power*cleft
    powerRight = power*cright

    # accumulate onto spectrum
    spectrum = np.array( [ 
        np.sum( powerLeft*(ileft ==i) + powerRight*(iright==i) )
        for i in range(len(kmid))] )

    return spectrum

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

    f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]*convfact
    f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]*convfact
    f01m = ad['flash',baseFlash+suffixFlash[2]+energyGroup]*convfact
    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]*convfact


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

    f00  = ad['flash',baseFlash+suffixFlash[0]+energyGroup]
    f11  = ad['flash',baseFlash+suffixFlash[1]+energyGroup]
    f01m = ad['flash',baseFlash+suffixFlash[2]+energyGroup]
    f01p = ad['flash',baseFlash+suffixFlash[3]+energyGroup]

    #fluences:

    f00 = f00*NR[0][0]
    f11 = f11*NR[1][1]

    Nm = np.sqrt(NR[0][1]**2 + NI[0][1]**2)
    Np = np.arctan2(-NI[0][1], NR[0][1])

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

def averaged_traceM(M, num_cells):

    sumtrace = 0.0
    for fi in range(NF):
        sumtrace += np.sum(M[fi][fi])
    return sumtrace/float(num_cells)

def averaged_Mcomp(Mcomp, num_cells):

    # do the averaging
    Mout = np.sum(Mcomp)/float(num_cells)
    return Mout

def averaged_Mmod(MR, MI, num_cells):

    # do the averaging
    Mout = np.sum(np.sqrt(MR**2 + MI**2))/float(num_cells)
    return Mout

def averaged_Mphi(MR, MI, num_cells):

    # do the averaging
    # note minus sign on NI:
    Mout = np.sum(np.arctan2(-MI, MR))/float(num_cells)
    return Mout

def variance_Mcomp(Mcomp, M_comp_avg, num_cells):

    # do the averaging
    Mout = np.sum((Mcomp-M_comp_avg)**2)/float(num_cells)
    return Mout

def variance_Mmod(MR, MI, M_mod_avg, num_cells):

    # do the averaging
    Mout = np.sum((np.sqrt(MR**2 + MI**2) - M_mod_avg)**2)/float(num_cells)
    return Mout

def variance_Mphi(MR, MI, M_phi_avg, num_cells):

    # do the averaging
    Mout = np.sum((np.arctan2(-MI, MR) - M_phi_avg)**2)/float(num_cells)
    return Mout

def correlation_MRI(MR, MI, MR_avg, MI_avg, MR_var, MI_var, num_cells):

    if MR_var != 0.0 and MI_var != 0.0:
        r_corr = np.sum((MR - MR_avg)*(MI - MI_avg))/ \
            float(num_cells)/np.sqrt(MR_var*MI_var)
    else:
        r_corr = 0.0

    return r_corr

def correlation_Mmp(MR, MI, Mm_avg, Mp_avg, Mm_var, Mp_var, num_cells):

    if Mm_var != 0.0 and Mp_var != 0.0:
        r_corr = np.sum((np.sqrt(MR**2 + MI**2) - Mm_avg)*(np.arctan2(-MI, MR) - Mp_avg))/ \
            float(num_cells)/np.sqrt(Mm_var*Mp_var)
    else:
        r_corr = 0.0

    return r_corr




#########################
# loop over directories #
#########################
if do_MPI:
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()
else:
    mpi_rank = 0
    mpi_size = 1
directories = sorted(glob.glob(output_base+"*"))
if( (not do_average) and (not do_fft)):
    directories = []
for d in directories[mpi_rank::mpi_size]:
    print("# rank",mpi_rank,"is working on", d)
    sys.stdout.flush()
    eds = emu.EmuDataset(d)
    t = eds.ds.current_time
    ad = eds.ds.all_data()

    ################
    # average work #
    ################
    # write averaged data
    outputfilename = "reduced_data_var_"+d
    already_done = len(glob.glob(outputfilename))>0
    if do_average and not already_done:
        thisN, thisNI = get_matrix_N("")
        num_cells = np.product(np.shape(thisN[0][0]))

        N_trace_avg = averaged_traceM(thisN, num_cells)

        N_real_avg = np.zeros([NF,NF])
        N_imag_avg = np.zeros([NF,NF])
        N_mod_avg = np.zeros([NF,NF])
        N_phi_avg = np.zeros([NF,NF])
        for i in range(NF):
            for j in range(i,NF):
                N_real_avg[i][j] = averaged_Mcomp(thisN[i][j],num_cells)
        N_real_avg[1][0] = N_real_avg[0][1]
        N_imag_avg[0][1] = averaged_Mcomp(thisNI[0][1],num_cells)
        N_imag_avg[1][0] = -N_imag_avg[0][1]
        N_mod_avg[0][1] = averaged_Mmod(thisN[0][1],thisNI[0][1],num_cells)
        N_mod_avg[1][0] = N_mod_avg[0][1]
        N_mod_avg[0][0] = N_real_avg[0][0]
        N_mod_avg[1][1] = N_real_avg[1][1]
        N_phi_avg[0][1] = averaged_Mphi(thisN[0][1],thisNI[0][1], num_cells)
        N_phi_avg[1][0] = -N_phi_avg[0][1]

        N_real_var = np.zeros([NF,NF])
        N_imag_var = np.zeros([NF,NF])
        N_mod_var = np.zeros([NF,NF])
        N_phi_var = np.zeros([NF,NF])
        for i in range(NF):
            for j in range(i,NF):
                N_real_var[i][j] = variance_Mcomp(thisN[i][j], N_real_avg[i][j], num_cells)
        N_real_var[1][0] = N_real_var[0][1]
        N_imag_var[0][1] = variance_Mcomp(thisNI[0][1], N_imag_avg[0][1], num_cells)
        N_imag_var[1][0] = N_imag_var[0][1]
        N_mod_var[0][1] = variance_Mmod(thisN[0][1], thisNI[0][1], N_mod_avg[0][1], num_cells)
        N_mod_var[1][0] = N_mod_var[0][1]
        N_mod_var[0][0] = N_real_var[0][0]
        N_mod_var[1][1] = N_real_var[1][1]
        N_phi_var[0][1] = variance_Mphi(thisN[0][1], thisNI[0][1], N_phi_avg[0][1], num_cells)
        N_phi_var[1][0] = N_phi_var[0][1]

        N_corr_RI = correlation_MRI(thisN[0][1], thisNI[0][1], N_real_avg[0][1], N_imag_avg[0][1], N_real_var[0][1], N_imag_var[0][1], num_cells)
        N_corr_mp = correlation_Mmp(thisN[0][1], thisNI[0][1], N_mod_avg[0][1], N_phi_avg[0][1], N_mod_var[0][1], N_phi_var[0][1], num_cells)


        thisFx, thisFxI = get_matrix_F("Fx","", thisN, thisNI)
        thisFy, thisFyI = get_matrix_F("Fy","", thisN, thisNI)
        thisFz, thisFzI = get_matrix_F("Fz","", thisN, thisNI)
        Ftmp  = np.array([thisFx , thisFy , thisFz ])
        FtmpI = np.array([thisFxI, thisFyI, thisFzI])

        F_real_avg = np.zeros([3,NF,NF])
        F_imag_avg = np.zeros([3,NF,NF])
        F_mod_avg = np.zeros([3,NF,NF])
        F_phi_avg = np.zeros([3,NF,NF])
        for i in range(3):
            for j in range(NF):
                for k in range(j,NF):
                    F_real_avg[i,j,k] = averaged_Mcomp(Ftmp[i,j,k], num_cells)
            F_imag_avg[i,0,1] = averaged_Mcomp(FtmpI[i,0,1], num_cells)
            F_mod_avg[i,0,1] = averaged_Mmod(Ftmp[i,0,1], FtmpI[i,0,1], num_cells)
            F_mod_avg[i,0,0] = averaged_Mcomp(np.abs(Ftmp[i,0,0]), num_cells)
            F_mod_avg[i,1,1] = averaged_Mcomp(np.abs(Ftmp[i,1,1]), num_cells)
            F_phi_avg[i,0,1] = averaged_Mphi(Ftmp[i,0,1], FtmpI[i,0,1], num_cells)
        F_real_avg[:,1,0] = F_real_avg[:,0,1]
        F_imag_avg[:,1,0] = -F_imag_avg[:,0,1]
        F_mod_avg[:,1,0] = F_mod_avg[:,0,1]
        F_phi_avg[:,1,0] = -F_phi_avg[:,0,1]

        F_real_var = np.zeros([3,NF,NF])
        F_imag_var = np.zeros([3,NF,NF])
        F_mod_var = np.zeros([3,NF,NF])
        F_phi_var = np.zeros([3,NF,NF])
        for i in range(3):
            for j in range(NF):
                for k in range(j,NF):
                    F_real_var[i,j,k] = variance_Mcomp(Ftmp[i,j,k], F_real_avg[i,j,k], num_cells)
            F_imag_var[i,0,1] = variance_Mcomp(FtmpI[i,0,1], F_imag_avg[i,0,1], num_cells)
            F_mod_var[i,0,1] = variance_Mmod(Ftmp[i,0,1], FtmpI[i,0,1], F_mod_avg[i,0,1], num_cells)
            F_mod_var[i,0,0] = variance_Mcomp(np.abs(Ftmp[i,0,0]), F_mod_avg[i,0,0], num_cells)
            F_mod_var[i,1,1] = variance_Mcomp(np.abs(Ftmp[i,1,1]), F_mod_avg[i,1,1], num_cells)
            F_phi_var[i,0,1] = variance_Mphi(Ftmp[i,0,1], FtmpI[i,0,1], F_phi_avg[i,0,1], num_cells)
        F_real_var[:,1,0] = F_real_var[:,0,1]
        F_imag_var[:,1,0] = F_imag_var[:,0,1]
        F_mod_var[:,1,0] = F_mod_var[:,0,1]
        F_phi_var[:,1,0] = F_phi_var[:,0,1]

        F_corr_RI = np.zeros([3])
        F_corr_mp = np.zeros([3])
        for i in range(3):
            F_corr_RI[i] = correlation_MRI(Ftmp[i,0,1], FtmpI[i,0,1], F_real_avg[i,0,1], F_imag_avg[i,0,1], \
                    F_real_var[i,0,1], F_imag_var[i,0,1], num_cells)
            F_corr_mp[i] = correlation_Mmp(Ftmp[i,0,1], FtmpI[i,0,1], F_mod_avg[i,0,1], F_phi_avg[i,0,1], \
                    F_mod_var[i,0,1], F_phi_var[i,0,1], num_cells)

        #thisN, thisNI = get_matrix_N("bar")
        #sumtrace = sumtrace_N(thisN)
        ##sumtrace = sumtrace_N(thisN)*NF_2_to_3_bnu
        #sumtrace = 1.0
        #Nbar = averaged_N(thisN,thisNI,sumtrace)

        #thisFx, thisFxI = get_matrix_F("Fx","bar", thisN, thisNI)
        #thisFy, thisFyI = get_matrix_F("Fy","bar", thisN, thisNI)
        #thisFz, thisFzI = get_matrix_F("Fz","bar", thisN, thisNI)
        #Ftmp  = np.array([thisFx , thisFy , thisFz ])
        #FtmpI = np.array([thisFxI, thisFyI, thisFzI])
        #Fbar = averaged_F(Ftmp, FtmpI,sumtrace)

        print("# rank",mpi_rank,"writing",outputfilename)
        sys.stdout.flush()
        avgData = h5py.File(outputfilename,"w")
        avgData["N_avg_real"] = [N_real_avg/N_trace_avg,]
        avgData["N_avg_imag"] = [N_imag_avg/N_trace_avg,]
        avgData["N_avg_mag"] = [N_mod_avg/N_trace_avg,]
        avgData["N_avg_phase"] = [N_phi_avg,]
        avgData["N_var_real"] = [N_real_var/N_trace_avg**2,]
        avgData["N_var_imag"] = [N_imag_var/N_trace_avg**2,]
        avgData["N_var_mag"] = [N_mod_var/N_trace_avg**2,]
        avgData["N_var_phase"] = [N_phi_var,]
        avgData["N_corr_comp"] = [N_corr_RI,]
        avgData["N_corr_pol"] = [N_corr_mp,]
        avgData["F_avg_real"] = [F_real_avg/N_trace_avg,]
        avgData["F_avg_imag"] = [F_imag_avg/N_trace_avg,]
        avgData["F_avg_mag"] = [F_mod_avg/N_trace_avg,]
        avgData["F_avg_phase"] = [F_phi_avg,]
        avgData["F_var_real"] = [F_real_var/N_trace_avg**2,]
        avgData["F_var_imag"] = [F_imag_var/N_trace_avg**2,]
        avgData["F_var_mag"] = [F_mod_var/N_trace_avg**2,]
        avgData["F_var_phase"] = [F_phi_var,]
        avgData["F_corr_comp"] = [F_corr_RI,]
        avgData["F_corr_pol"] = [F_corr_mp,]
        #avgData["Nbar_avg_mag"] = [Nbar,]
        #avgData["F_avg_mag"] = [F,]
        #avgData["Fbar_avg_mag"] = [Fbar,]
        avgData["t"] = [t,]
        avgData.close()

    ############
    # FFT work #
    ############
    outputfilename = "reduced_data_fft_power_"+d
    already_done = len(glob.glob(outputfilename))>0
    if do_fft and not already_done:

        print("# rank",mpi_rank,"writing",outputfilename)
        sys.stdout.flush()
        fout = h5py.File(outputfilename,"w")
        fout["t"] = [np.array(t),]

        fft = eds.fourier("ee"+energyGroup,nproc=nproc)
        fout["k"] = get_kmid(fft)
        cleft, cright, ileft, iright, kmid = fft_coefficients(fft)
        N00_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
        fft = eds.fourier("ea"+energyGroup,nproc=nproc)
        N11_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
        fft = eds.fourier("er"+energyGroup,"ep"+energyGroup,nproc=nproc)
        N01_FFT = fft_power(fft, cleft, cright, ileft, iright, kmid)
        fout["N00_FFT"] = [np.array(N00_FFT),]
        fout["N11_FFT"] = [np.array(N11_FFT),]
        fout["N01_FFT"] = [np.array(N01_FFT),]
        
        fout.close()

