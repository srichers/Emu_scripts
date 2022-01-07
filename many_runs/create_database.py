import copy
import h5py
import glob
import re
import numpy as np

# INPUTS
model_filename = "../../orthonormal_distributions/model_rl0_orthonormal.h5"


# get data from input parameters
model = h5py.File(model_filename,"r")
descriminant = np.array(model["crossing_descriminant"])
model.close()

# get the list of files to process
flist = sorted(glob.glob("*/reduced_data.h5"))
nsims = len(flist)
print()
print(str(len(flist)) + " files")

# get the number of flavors
data = h5py.File(flist[0],"r")
nf = np.shape(np.array(data["N_avg_mag"]))[1]
data.close()
print(str(nf) + " flavors")

# set output arrays to zero
iList = np.zeros(nsims, dtype=int)
jList = np.zeros(nsims, dtype=int)
kList = np.zeros(nsims, dtype=int)
maxOffDiagList = np.zeros((nsims, 2))
minOffDiagList = np.zeros((nsims, 2))
growthRateList = np.zeros((nsims,2))
tSaturationList = np.zeros((nsims,2))
NFinalnpoints = np.zeros((nsims,2), dtype=int)
NFinalList = np.zeros((nsims, 2, nf))
NInitialList = np.zeros((nsims, 2, nf))
crossingDescriminantList = np.zeros(nsims)

def calculate_offdiag_mag(N, Nbar):
    N = np.array(N)
    Nbar = np.array(Nbar)
    nt = N.shape[0]
    Noffdiag = np.zeros((nt,2))
    Ndiag = np.zeros((nt,2,nf))
    for f1 in range(nf):
        Ndiag[:,0,f1] =    N[:,f1,f1]
        Ndiag[:,1,f1] = Nbar[:,f1,f1]
        for f2 in range(f1+1, nf):
            Noffdiag[:,0] +=    N[:,f1,f2]**2
            Noffdiag[:,1] += Nbar[:,f1,f2]**2
    return Ndiag, np.sqrt(Noffdiag)


for f,fi in zip(flist, range(len(flist))):
    # get global simulation coordinates
    ijk = re.findall(r'[0-9]{3}', f)
    i = int(ijk[0])
    j = int(ijk[1])
    k = int(ijk[2])
    iList[fi] = i
    jList[fi] = j
    kList[fi] = k

    # calculate the crossing descriminant
    crossingDescriminantList[fi] = descriminant[i,j,k]

    # read the data
    data = h5py.File(f,"r")
    t = np.array(data["t"]) # seconds
    Ndiag, offdiag_mag = calculate_offdiag_mag(data["N_avg_mag"], data["Nbar_avg_mag"]) # trace-normalized
    data.close()

    for im in range(2):
        # get the amplification factor of the off-diagonal magnitude
        minOffDiag = np.min(offdiag_mag[:,im], axis=0)
        maxOffDiag = np.max(offdiag_mag[:,im], axis=0)
        minOffDiagList[fi,im] = minOffDiag
        maxOffDiagList[fi,im] = maxOffDiag
 
        # get the indexes from which to calculate growth rate
        i0 = np.where(offdiag_mag > 10.*minOffDiag)[0]
        i1 = np.where(offdiag_mag < 0.1*maxOffDiag)[0]
        if len(i0)>0 and len(i1)>0:
            i0 = i0[0]
            i1 = i1[-1]
        else:
            i0=0
            i1=0
        isGrowing = i1>i0

        # do the following if we detect rapid growth
        if isGrowing:

            # calculate the growth rate
            growthRate = np.log(offdiag_mag[i1,im]/offdiag_mag[i0,im]) / (t[i1]-t[i0])
            growthRateList[fi,im] = growthRate

            # calculate the time of saturation
            isaturation = np.argmax(offdiag_mag[:,im])
            isSaturated = (isaturation < len(t)-1)
            if isSaturated:
                tSaturation = t[isaturation]
                tSaturationList[fi,im] = tSaturation

                # calculate final abundances
                integrableTimes = np.where(t>3.*tSaturation)[0]
                npoints = len(integrableTimes)
                if npoints>0 :
                    iStart = integrableTimes[0]
                    NFinal = np.sum(Ndiag[iStart:,im,:], axis=0) / npoints
                    NFinalList[fi,im,:] = NFinal
                    NInitialList[fi,im,:] = Ndiag[0,im,:]
                    NFinalnpoints[fi,im] = npoints



# output some numbers
def printStat(a, tail):
    ablist = [len(np.where(a[:,0]>0)[0]), len(np.where(a[:,1]>0)[0])]
    print(str(ablist[0])+"/"+str(ablist[1])+" simulations "+tail+" (nu/antinu)")

hasCrossing = np.zeros_like(crossingDescriminantList)
hasCrossing[np.where(crossingDescriminantList>=0)] = 1

ncross = np.sum(hasCrossing)
print(str(ncross) + " have an ELN crossing")

def minMaxAvg(A, include):
    minA = np.zeros(2)
    maxA = np.zeros(2)
    avgA = np.zeros(2)

    for im in range(2):
        Atmp = A[:,im][np.where(include[:,im]==True)]
        minA[im] = np.min(Atmp)
        maxA[im] = np.max(Atmp)
        avgA[im] = np.mean(Atmp)
    return minA, maxA, avgA

minGR, maxGR, avgGR = minMaxAvg(growthRateList, growthRateList>0)
minDeltaNe, maxDeltaNe, avgDeltaNe = minMaxAvg(NFinalList[:,:,0] - NInitialList[:,:,0],growthRateList>0)

printStat(growthRateList, "are growing quickly")
printStat(growthRateList*hasCrossing[:,np.newaxis], "are growing quickly and have an ELN crossing")
printStat(tSaturationList, "saturated")
printStat(NFinalnpoints, "are past 3x saturation time")
print()
print(minGR,"minimum growth rate")
print(maxGR,"maximum growth rate")
print(avgGR,"average growth rate")
print()
print(minDeltaNe,"minimum delta fraction nu_e")
print(maxDeltaNe,"maximum delta fraction nu_e")
print(avgDeltaNe,"average delta fraction nu_e")

minNe = np.zeros(2)
maxNe = np.zeros(2)
avgNe = np.zeros(2)

# Write data to hdf5 file
output = h5py.File("many_sims_database.h5","w")
output["nf"] = nf
output["i"] = iList
output["j"] = jList
output["k"] = kList
output["minOffDiag_trace"] = minOffDiagList
output["maxOffDiag_trace"] = maxOffDiagList
output["growthRate(1|s)"] = growthRateList
output["tSaturation(s)"] = tSaturationList
output["Nfinal_trace"] = NFinalList
output["Ninitial_trace"] = NInitialList
output["Nfinalnpoints"] = NFinalnpoints
output.close()
print()
