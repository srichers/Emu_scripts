import numpy as np
import h5py
import sys
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import shutil

infilename = "../../orthonormal_distributions/model_rl0_orthonormal.h5"
slicedir = 'none' # 'x', 'y', 'z', 'none'
sliceind = 100
stepsize = 4
run_directory = "../RUN_lowres_sqrt2"
reference_domainsize = 16.0 / np.sqrt(2.) # cm
reference_phi0 = 4.89e32 * 2.    # 1/ccm
reference_phi1 = 4.89e32 * 2./3. # 1/ccm
exist_ok = False
do_only_crossings = True
only_counting = False

#############
# constants #
#############
hbar = 1.05457266e-27 # erg s
c = 2.99792458e10 # cm/s
eV = 1.60218e-12 # erg
MeV = 1e6 * eV   # erg
GeV = 1e9 * eV   # erg
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3

################
# DATA READING #
################
def read_dataset(infile, dsetname):
    full_dataset = np.array(infile[dsetname])
    #if slicedir=='x':
    #    return full_dataset[sliceind,:,:]
    #elif slicedir=='y':
    #    return full_dataset[:,sliceind,:]
    #elif slicedir=='z':
    #    return full_dataset[:,:,sliceind]
    #else:
    return full_dataset

infile = h5py.File(infilename,"r")
x = read_dataset(infile,"x(cm)")
y = read_dataset(infile,"y(cm)")
z = read_dataset(infile,"z(cm)")
J_e = read_dataset(infile,"J_e(erg|ccm)")
J_a = read_dataset(infile,"J_a(erg|ccm)")
J_x = read_dataset(infile,"J_x(erg|ccm)")
n_e = read_dataset(infile,"n_e(1|ccm)")
n_a = read_dataset(infile,"n_a(1|ccm)")
n_x = read_dataset(infile,"n_x(1|ccm)")
fn_e = read_dataset(infile,"fn_e(1|ccm)")
fn_a = read_dataset(infile,"fn_a(1|ccm)")
fn_x = read_dataset(infile,"fn_x(1|ccm)")
fluxfac_e = read_dataset(infile,"fluxfac_e(1|ccm)")
fluxfac_a = read_dataset(infile,"fluxfac_a(1|ccm)")
fluxfac_x = read_dataset(infile,"fluxfac_x(1|ccm)")
eddfac_e = read_dataset(infile,"eddfac_e(1|ccm)")
eddfac_a = read_dataset(infile,"eddfac_a(1|ccm)")
eddfac_x = read_dataset(infile,"eddfac_x(1|ccm)")
Z_e = read_dataset(infile,"minerbo_Ze")
Z_a = read_dataset(infile,"minerbo_Za")
Z_x = read_dataset(infile,"minerbo_Zx")
descriminant = read_dataset(infile,"crossing_descriminant")
rho = read_dataset(infile,"rho(g|ccm)")
Ye = read_dataset(infile,"Ye")
infile.close()
nnet = n_e+n_a+n_x
Jnet = J_e+J_a+J_x
print("There are",len(np.where(descriminant>0)[0]),"data points with crossings.")

# get list of indices to loop over
datashape = np.shape(J_e)
print("data shape = ", datashape)
ixlist = range(0, datashape[0], stepsize)
iylist = range(0, datashape[1], stepsize)
izlist = range(0, datashape[2], stepsize)
if slicedir=='x':
    ixlist = [sliceind,]
if slicedir=='y':
    iylist = [sliceind,]
if slicedir=='z':
    izlist = [sliceind,]

# determine the total SI potential (x flavors are identical, so they dont enter)
# eln is already aligned with the z axis
phi0 = n_e - n_a # + n_electron ?
phi1 = fn_e[2] - fn_a[2]
SI_strength = np.abs(phi0) # + phi1)

# make runs directory
copy_list = ["inputs_sample", "setup_runs.py", "reduce_data.py", "emu_yt_module.py", "amrex_plot_tools.py", "multiple_nodes_many_tasks_parallel.sh", "payload.sh", "task.sh","combine_files.py","convertToHDF5.py"]

if not only_counting:
    os.makedirs(run_directory, exist_ok=exist_ok)
    for filename in copy_list:
        shutil.copy(filename,run_directory+"/"+filename)
    shutil.copy("../main3d.gnu.haswell.TPROF.ex", run_directory+"/main3d.gnu.haswell.TPROF.ex")

# set up each simulation within run_directory
count = 0
for i in ixlist:
    for j in iylist:
        for k in izlist:
            count += 1

            # don't set up the simulation if there is no crossing
            if (do_only_crossings and descriminant[i,j,k]<0) or only_counting:
                continue
            
            dirname = run_directory+"/i"+str(i).zfill(3)+"/j"+str(j).zfill(3)+"/k"+str(k).zfill(3)
            
            # create the simulation directory
            os.makedirs(dirname, exist_ok=exist_ok)

            # copy parameter file to destination
            shutil.copyfile("inputs_sample",dirname+"/inputs")
            
            # append our parameters to it
            inputs = open(dirname+"/inputs","a")
            inputs.write("st5_nnue = "+ str(n_e[i,j,k]) + "\n")
            inputs.write("st5_nnua = "+ str(n_a[i,j,k]) + "\n")
            inputs.write("st5_nnux = "+ str(n_x[i,j,k]) + "\n")
            inputs.write("st5_fxnue = "+ str(fn_e[0,i,j,k] / n_e[i,j,k]) + "\n")
            inputs.write("st5_fynue = "+ str(fn_e[1,i,j,k] / n_e[i,j,k]) + "\n")
            inputs.write("st5_fznue = "+ str(fn_e[2,i,j,k] / n_e[i,j,k]) + "\n")
            inputs.write("st5_fxnua = "+ str(fn_a[0,i,j,k] / n_a[i,j,k]) + "\n")
            inputs.write("st5_fynua = "+ str(fn_a[1,i,j,k] / n_a[i,j,k]) + "\n")
            inputs.write("st5_fznua = "+ str(fn_a[2,i,j,k] / n_a[i,j,k]) + "\n")
            inputs.write("st5_fxnux = "+ str(fn_x[0,i,j,k] / n_x[i,j,k]) + "\n")
            inputs.write("st5_fynux = "+ str(fn_x[1,i,j,k] / n_x[i,j,k]) + "\n")
            inputs.write("st5_fznux = "+ str(fn_x[2,i,j,k] / n_x[i,j,k]) + "\n")
            inputs.write("st5_avgE_MeV ="+str(Jnet[i,j,k] / nnet[i,j,k] / MeV) + "\n")
            inputs.write("rho_g_ccm = "+ str(rho[i,j,k]) + "\n")
            inputs.write("Ye = " + str(Ye[i,j,k]) + "\n")
            inputs.write("Lz = "+ str(reference_domainsize * reference_phi1 / phi1[i,j,k]) + "\n")
            inputs.close()

print("Set up",count,"simulations.")

i = 50
j = 100
k = 21
print(i, j, k, phi0[i,j,k], phi1[i,j,k], SI_strength[i,j,k], reference_domainsize * reference_phi1 / phi1[i,j,k])
kpredicted = reference_phi1 * GF / (hbar*c)
print(2.*np.pi/kpredicted)
