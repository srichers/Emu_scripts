import numpy as np
import matplotlib.pyplot as plt
import glob
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from matplotlib.lines import Line2D

variables=["N","Fx","Fy","Fz"]
flavors=["00","11","22","01","02","12"]

#Variable(s) to plot (only specify here if you are only plotting one variable, otherwise use a loop)
variable = variables[0]
flavor = flavors[3]

#snapshots to use: pre-saturation, post-saturation, more post-saturation
snapshots = [0.2e-09,1e-09,2e-09]

################
# plot options #
################
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
#mpl.rc('text', usetex=True)
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

fig, axes = plt.subplots(3,3, figsize=(16,10), sharex = True, sharey = True)
plt.subplots_adjust(hspace=0,wspace=0.08)

##############
# formatting #
##############
for ax in axes.flat:
    ax.set(xlabel=r"$k\,(\mathrm{cm}^{-1})$", ylabel=r"$|\widetilde{f}|^2\,(\mathrm{cm}^{-2})$", xlim=(0,8),ylim=(1e-22,1e-0))
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes.flat:
    ax.label_outer()

#axes[0].set_xticklabels([])
axes[0,0].text(.05,.85,"Fiducial", transform=axes[0,0].transAxes)
axes[1,0].text(.05,.85,"90 Degree",transform=axes[1,0].transAxes)
axes[2,0].text(.05,.85,"2/3",transform=axes[2,0].transAxes)
for i in range(len(snapshots)):
    axes[0,i].text(.6,.85,"t=%0.1f ns"%(snapshots[i]*1e9), transform=axes[0,i].transAxes)

def plotdata(v,f,ax,t,c,l,lab,filename):
    # get appropriate data
    data = h5py.File(filename,"r")
    times=data["t"]
    k=data["k"]
    fft = data[v+f+"_FFT"]
    total_power = np.sum(fft)
    #snapshots to use: pre-saturation, during saturation, post-saturation
    #snapshots = [0.2e-09,1e-09,2e-09]
    #this will plot an fft line at the timestep closest to each snapshot specified above
    #for s in range(len(snapshots)):
    s_index = (np.abs(times[:] - float(t))).argmin()
    ax.semilogy(k[:], fft[s_index,:-1]/total_power, color=c,linestyle=l,label=lab)
    
    data.close()

#############
# plot data #
#############
dirlist    = [["gray" ,"1D", "global/project/projectdirs/m3018/Emu/PAPER/1D/converge_direction/1D_dir64"],
              ["black","2D", "ocean/projects/phy200048p/shared/2D/manydirections64"],
              ["blue" ,"3D", "global/project/projectdirs/m3761/3D/128r64d_128n800s16mpi"]]
for inputs in dirlist:
    filename = inputs[2]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[0,ind],val,inputs[0],'-',inputs[1],filename)

dirlist    = [["gray" ,"-" ,"1D", "global/project/projectdirs/m3018/Emu/PAPER/1D/fbar_direction/90"],
              ["black","-" ,"2D (in plane)", "ocean/projects/phy200048p/shared/2D/90deg_inplane/fft"],
              ["black","--","2D (out of plane)", "ocean/projects/phy200048p/shared/2D/90deg_outofplane/fft"],
              ["blue","-","3D","global/project/projectdirs/m3761/3D/90degree_64d"]]
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[1,ind],val,inputs[0],inputs[1],inputs[2],filename)

dirlist    = [["gray", "1D", "global/project/projectdirs/m3018/Emu/PAPER/1D/rando_test/1.0_thirds"],
              ["blue", "3D", "global/project/projectdirs/m3761/3D/two_thirds"]]
for inputs in dirlist:
    filename = inputs[2]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[2,ind],val,inputs[0],'-',inputs[1],filename)

#Create custom legend
axes[0,0].legend(frameon=False,fontsize=13, loc=(.6,.45))
axes[1,0].legend(frameon=False,fontsize=13, loc=(.45,.45))

############
# save pdf #
############
plt.savefig("power_spectrum.pdf", bbox_inches="tight")
