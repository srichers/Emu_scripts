# Given a full set of simulations sorted into i???/j???/k???/allData.h5, extract a database of the FFI growth rate and saturation properties
# Run from within a single RUN directory
import copy
import h5py
import glob
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# INPUTS
do_plot = True
growthrate_minfac = 1e-3
growthrate_maxfac = 1e-1

# get the list of files to process
dirlist = sorted(glob.glob("i*/j*/k*"))
nsims = len(dirlist)
print()
print(nsims,"files")

# get the number of flavors
data = h5py.File(dirlist[0]+"/allData.h5","r")
if "N22_Re" in data.keys(): nf = 3
else: nf = 2
data.close()
print(str(nf) + " flavors")

# set output arrays to zero
iList = np.zeros(nsims, dtype=int)
jList = np.zeros(nsims, dtype=int)
kList = np.zeros(nsims, dtype=int)
growthRateList = np.zeros(nsims)
F4FinalList   = np.zeros((nsims, 4, 2, nf))
F4InitialList = np.zeros((nsims, 4, 2, nf))
F4FinalStddevList = np.zeros((nsims, 4, 2, nf))
#==============#
# plot options #
#==============#
if(do_plot):
    mpl.rcParams['font.size'] = 22
    mpl.rcParams['font.family'] = 'serif'
    mpl.rc('text', usetex=True)
    mpl.rcParams['xtick.major.size'] = 7
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['xtick.major.pad'] = 8
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.minor.width'] = 2
    mpl.rcParams['ytick.major.size'] = 7
    mpl.rcParams['ytick.major.width'] = 2
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.minor.width'] = 2
    mpl.rcParams['axes.linewidth'] = 2
    fig,axes=plt.subplots(1,3, figsize=(16,8))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.align_labels()

#========================#
# get the grid cell size #
#========================#
def cell_size(dirname):
    f = open(dirname+"/inputs","r")
    lines = f.readlines()
    for line in lines:
        if "Lz" in line:
            Lz = float(line.split("=")[1])
        if "ncell" in line:
            nz = int(re.split("[, \)]", line)[-2])
    f.close()
    return Lz/nz

#==========================#
# get the growth rate, etc #
#==========================#
def growth_properties(t, data):
    # just use antineutrinos because I'm stupid and my data-collapsing script ignored 01_Im quantities
    Nbar_offdiag_mag = np.sqrt( np.array(data["N01_Rebar"])**2 + np.array(data["N01_Imbar"])**2) # [it, nz]
    Nbar_offdiag_mag = np.average(Nbar_offdiag_mag, axis=1) # [it]

    # get the time of the peak
    imax = np.argmax(Nbar_offdiag_mag)

    # get the growth rate
    minOffDiag = Nbar_offdiag_mag[0] #np.min(Nbar_offdiag_mag)
    maxOffDiag = np.max(Nbar_offdiag_mag)

    # calculate the growth rate
    growthRate = 0
    locs  = np.where((Nbar_offdiag_mag > growthrate_minfac*maxOffDiag) & (Nbar_offdiag_mag < growthrate_maxfac*maxOffDiag) & (t<t[imax]))[0]
    if len(locs)>1:
        i0 = locs[0]
        i1 = locs[-1]
        growthRate = np.log(Nbar_offdiag_mag[i1]/Nbar_offdiag_mag[i0]) / (t[i1]-t[i0])

        if(t[-1]>3*t[imax] and do_plot):
            axes[0].semilogy(t/t[imax], Nbar_offdiag_mag/maxOffDiag, color="gray")
            axes[0].scatter([t[i1]/t[imax]], [Nbar_offdiag_mag[i1]/maxOffDiag], color="gray")
            axes[0].scatter([t[i0]/t[imax]], [Nbar_offdiag_mag[i0]/maxOffDiag], color="gray")
            
    return imax, growthRate

#===========================================#
# get the final moments of the distribution #
#===========================================#
# divide by dz because dx and dy are assumed to be 1
def final_properties(imax, t, growthRate, data, dz):
    nt,nz = np.shape(np.array(data["N00_Re"]))
    
    # [xyzt, nu/antinu, flavor, it]
    F4 = np.zeros((4,2,nf,nt,nz))
    for ixyzt, xyzt in zip(range(4),["Fx","Fy","Fz","N"]):
        for isuffix, suffix in zip(range(2),["","bar"]):
            for flavor in range(nf):
                F4[ixyzt,isuffix,flavor,:,:] = np.array(data[xyzt+str(flavor)+str(flavor)+"_Re"+suffix]) / dz

    # get the time interval over which to average
    F4_initial = np.average(F4[:,:,:,0,:], axis=3)
    F4_final = np.zeros((4,2,nf))
    F4_final_stddev = np.zeros_like(F4_final)
    tmax = t[imax]
    tend = t[-1]
    if tend > 3*tmax and growthRate>0:
        i0 = np.argmin(np.abs(t-2*tmax))
        F4_final[:,:,:]        = np.average(F4[:,:,:,i0:-1,:], axis=(3,4))
        F4_final_stddev[:,:,:] = np.std(    F4[:,:,:,i0:-1,:], axis=(3,4))

        if do_plot:
            Navg = np.average(F4[3,:,0,:,:], axis=2)
            axes[1].plot(t/tmax, Navg[0,:]/Navg[0,0], color="gray")
            axes[2].plot(t/tmax, Navg[1,:]/Navg[1,0], color="gray")

    return F4_initial, F4_final, F4_final_stddev

#=================#
# LOOP OVER FILES #
#=================#
for d,di in zip(dirlist, range(nsims)):
    print(d)
    # get the cell size
    dz = cell_size(d)
    
    # get global simulation coordinates
    ijk = re.findall(r'[0-9]{3}', d)
    i = int(ijk[0])
    j = int(ijk[1])
    k = int(ijk[2])
    iList[di] = i
    jList[di] = j
    kList[di] = k

    # open the data file
    data = h5py.File(d+"/allData.h5","r")
    t = np.array(data["t(s)"]) # seconds

    # get the growth rate
    imax, growthRate = growth_properties(t,data)
    growthRateList[di] = growthRate
    
    # get the final state
    F4_initial, F4_final, F4_final_stddev = final_properties(imax, t, growthRate, data, dz)
    F4InitialList[di] = F4_initial
    F4FinalList[di] = F4_final
    F4FinalStddevList[di] = F4_final_stddev
    
    # close the data file
    data.close()


if do_plot:
    ax2 = axes[2].twinx()
    ax2.tick_params(axis='both',which="both", direction="in",top=True,right=True)
    ax2.minorticks_on()
    ax2.set_ylabel(r"$N/N(0)$")
    for ax in axes:
        ax.set_xlabel(r"$t/t_\mathrm{peak}$")
        ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
        ax.minorticks_on()
    for ax in axes[1:]:
        for label in ax.get_yticklabels():
            label.set_visible(False)
        ax.set_ylim(0,1)
    ax2.set_ylim(0,1)
    axes[0].set_ylabel(r"$\bar{N}_{e\mu}/\bar{N}_{e\mu,\mathrm{max}}$")
    plt.savefig("instability_comparison.pdf",bbox_inches='tight')
        
# Write data to hdf5 file
output = h5py.File("many_sims_database.h5","w")
output["nf"] = nf
output["i"] = iList
output["j"] = jList
output["k"] = kList
output["growthRate(1|s)"] = growthRateList
output["F4_final(1|ccm)"] = F4FinalList
output["F4_final_stddev(1|ccm)"] = F4FinalStddevList
output["F4_initial(1|ccm)"] = F4InitialList
output.close()
print()
