# Bring over the set of data common to two different datasets (presumably of different resolutions), but only if the data are resolved.
# run from the directory containing both RUN directories
import h5py
import re
import numpy as np
import random

directory_lo = "RUN_lowres_sqrt2"
directory_hi = "RUN_standard"
refinement_cellsize_factor = np.sqrt(2)
growthrate_tolerance = 0.01
F4_tolerance = 0.01
initial_ndens_tolerance = 1.2e-2

database_filename = "many_sims_database.h5"
inputs_database_filename = "../orthonormal_distributions/model_rl0_orthonormal.h5"
output_filename = "many_sims_database_"+directory_lo+"_"+directory_hi+".h5"

# is the lo or hi resolution index behind?
# check i, then j, then k
# assumes k is the fastest changing index
def which_behind(ind_lo, ind_hi):
    # check if we are past the end of the arrays
    if ind_lo>=len(ilo) or ind_hi>=len(ihi): return "end"
    
    # check i first
    if ilo[ind_lo] < ihi[ind_hi]: return "lo"
    if ilo[ind_lo] > ihi[ind_hi]: return "hi"

    # check j if i is equal
    if jlo[ind_lo] < jhi[ind_hi]: return "lo"
    if jlo[ind_lo] > jhi[ind_hi]: return "hi"

    # check k if j is equal
    if klo[ind_lo] < khi[ind_hi]: return "lo"
    if klo[ind_lo] > khi[ind_hi]: return "hi"

    # all three indices match
    return "match"
    

# find the next pair of indices that cause i, j, and k to match
# For the first iteration, assumes that ind_lo and ind_hi are -1
def next_match(ind_lo, ind_hi):
    # step both indices forward one
    ind_lo += 1
    ind_hi += 1

    # march forward until a match is found
    while True:
        behind = which_behind(ind_lo, ind_hi)
        if behind == "end":   break
        if behind == "match": break
        if behind == "lo": ind_lo += 1
        if behind == "hi": ind_hi += 1

    # check that the output makes sense
    if behind=="match":
        assert(ind_lo<len(ilo))
        assert(ind_hi<len(ihi))
        assert(ilo[ind_lo] == ihi[ind_hi])
        assert(jlo[ind_lo] == jhi[ind_hi])
        assert(klo[ind_lo] == khi[ind_hi])

    return ind_lo, ind_hi

def error(lo, hi):
    return np.abs(lo-hi) / (np.abs(lo)+np.abs(hi))

# open the two datasets
f_lo = h5py.File(directory_lo+"/"+database_filename,"r")
f_hi = h5py.File(directory_hi+"/"+database_filename,"r")

# read the index arrays
nf = np.array(f_hi["nf"])
ilo = np.array(f_lo["i"])
jlo = np.array(f_lo["j"])
klo = np.array(f_lo["k"])
ihi = np.array(f_hi["i"])
jhi = np.array(f_hi["j"])
khi = np.array(f_hi["k"])
growthrate_lo = np.array(f_lo["growthRate(1|s)"]) # [ind]
growthrate_hi = np.array(f_hi["growthRate(1|s)"])
F4_final_stddev_lo = np.array(f_lo["F4_final_stddev(1|ccm)"]) # [ind, xyzt, nu/antinu, flavor]
F4_final_stddev_hi = np.array(f_hi["F4_final_stddev(1|ccm)"])
F4_final_lo   = np.array(f_lo["F4_final(1|ccm)"])
F4_final_hi   = np.array(f_hi["F4_final(1|ccm)"])
F4_initial_lo = np.array(f_lo["F4_initial(1|ccm)"])
F4_initial_hi = np.array(f_hi["F4_initial(1|ccm)"])

# Normalize radiation moments to total neutrino number
Ntot_lo = np.sum(F4_initial_lo[:,3,:,:], axis=(1,2))
Ntot_hi = np.sum(F4_initial_hi[:,3,:,:], axis=(1,2))
maxerror = np.max(error(Ntot_lo, Ntot_hi))
print("Max error in net neutrino density:",maxerror)
assert(maxerror < initial_ndens_tolerance)
F4_final_stddev_lo /= Ntot_lo[:,np.newaxis,np.newaxis,np.newaxis]
F4_final_stddev_hi /= Ntot_hi[:,np.newaxis,np.newaxis,np.newaxis]
F4_final_lo /= Ntot_lo[:,np.newaxis,np.newaxis,np.newaxis]
F4_final_hi /= Ntot_hi[:,np.newaxis,np.newaxis,np.newaxis]
F4_initial_lo /= Ntot_lo[:,np.newaxis,np.newaxis,np.newaxis]
F4_initial_hi /= Ntot_hi[:,np.newaxis,np.newaxis,np.newaxis]


# close both input datasets
f_lo.close()
f_hi.close()

# Find first match
ind_lo = 0
ind_hi = 0
ind_lo, ind_hi = next_match(ind_lo, ind_hi)

# set up lists
#ijkList_growthrate = []
growthRateList = []
#ijkList_F4 = []
F4_initial_list = []
F4_final_list = []

# loop over all entries to get subset of data that is resolved
print("growthrate_lo min/max:",np.min(growthrate_lo), np.max(growthrate_lo))
print("growthrate_hi min/max:",np.min(growthrate_hi), np.max(growthrate_hi))
while ind_lo<len(ilo) and ind_hi<len(ihi): # len(ilo), len(ihi)
    gr_lo = growthrate_lo[ind_lo]
    gr_hi = growthrate_hi[ind_hi]
    
    # record growth rate if converged
    if growthrate_hi[ind_hi] > 0:
        errorval = error(growthrate_lo[ind_lo], growthrate_hi[ind_hi])
        if( errorval < growthrate_tolerance):
            #ijkList_growthrate.append((ihi[ind_hi], jhi[ind_hi], khi[ind_hi]))
            growthRateList.append(growthrate_hi[ind_hi])

    # Check that total number is conserved
    Nfinal   = np.sum(F4_final_hi[ind_hi,3])
    if Nfinal > 0:
        errorval = error(Nfinal, 1)
        print("Nfinal_error =",errorval)
        assert(errorval < 1e-3)
        
        # record initial and final state if converged
        maxerror = np.max(np.abs(F4_final_lo[ind_lo] - F4_final_hi[ind_hi]))
        if(maxerror < F4_tolerance):
            print()
            #print("ijk =",ihi[ind_hi],jhi[ind_hi],khi[ind_hi])
            print("Nee_values:",F4_final_lo[ind_lo,3,:,0],F4_final_hi[ind_lo,3,:,0])
            print("Nee_stddev:",F4_final_stddev_lo[ind_lo,3,:,0],F4_final_stddev_hi[ind_lo,3,:,0])
            print("maxerror: ",maxerror)
            #ijkList_F4.append((ihi[ind_hi], jhi[ind_hi], khi[ind_hi]))
            F4_initial_list.append(F4_initial_hi[ind_hi])
            F4_final_list.append(F4_final_hi[ind_hi])
    
    # get the next index
    ind_lo, ind_hi = next_match(ind_lo, ind_hi)

# add in locations that are not unstable (no crossing)
f = h5py.File(inputs_database_filename,"r")
dtheta = np.array(f["deltaCrossingAngle"])
stable_locs = np.where(dtheta != dtheta)
n_e = np.array(f["n_e(1|ccm)"])[stable_locs]
n_a = np.array(f["n_a(1|ccm)"])[stable_locs]
n_x = np.array(f["n_x(1|ccm)"])[stable_locs]
ntot = n_e+n_a+n_x
n_e /= ntot
n_a /= ntot
n_x /= ntot

fn_e = np.array(f["fn_e(1|ccm)"])
fn_a = np.array(f["fn_a(1|ccm)"])
fn_x = np.array(f["fn_x(1|ccm)"])
fn_e = np.transpose(np.array([fn_e[i][stable_locs] for i in range(3)] )) # [ind, xyz]
fn_a = np.transpose(np.array([fn_a[i][stable_locs] for i in range(3)] ))
fn_x = np.transpose(np.array([fn_x[i][stable_locs] for i in range(3)] ))
f.close()

nunstable = len(F4_final_list)
print("# simulations w/ resolved UNSTABLE final state:", nunstable)
nstable = len(n_e)
zeros = np.zeros_like(F4_final_list[0]) # [xyzt, nu/antinu, flavor]
F4_stable_list = np.zeros((nstable,4,2,nf))
F4_stable_list[:,1:,0,0]  = fn_e
F4_stable_list[:,1:,1,0]  = fn_a
F4_stable_list[:,1:,:,1:] = fn_x[:,:,np.newaxis,np.newaxis]
F4_stable_list[:,3,0,0] = n_e
F4_stable_list[:,3,1,0] = n_a
for fi in range(1,nf):
    F4_stable_list[:,3,:,fi] = n_x[:,np.newaxis]/4.

nsample = min(nstable, nunstable)
random.shuffle(F4_stable_list)
F4_stable_list = F4_stable_list[:nsample]
F4_initial_list.extend(F4_stable_list)
F4_final_list.extend(F4_stable_list)
    


print("# simulations (lo):", len(ilo))
print("# simulations (hi):", len(ihi))
print("# simulations w/ resolved growth rate:", len(growthRateList))
print("# simulations w/ resolved final state:", len(F4_final_list))

# write data to file
f_out = h5py.File(output_filename,"w")
#f_out["ijk_growthrate"] = np.array(ijkList_growthrate)
f_out["growthrate(1|s)"] = np.array(growthRateList)
#f_out["ijk_F4"] = np.array(ijkList_F4)
f_out["F4_initial_Nsum1"] = np.array(F4_initial_list)
f_out["F4_final_Nsum1"] = np.array(F4_final_list)
f_out.close()
