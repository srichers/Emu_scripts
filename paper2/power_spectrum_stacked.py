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

fig, axes = plt.subplots(3,3, figsize=(16,10), sharey = True)
plt.subplots_adjust(hspace=0,wspace=0.08)

##############
# formatting #
##############
for ax in axes.flat:
    ax.set(xlabel=r"$k\,(\mathrm{cm}^{-1})$", ylabel=r"$|\widetilde{f}|^2\,(\mathrm{cm}^{-2})$",ylim=(1e-22,1e-0))
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.minorticks_on()

for ax in axes[:,:2].flat:
    ax.set_xlim(0,8)
for ax in axes[:,2]:
    ax.set_xlim(0,2)

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axes.flat:
    ax.label_outer()

#axes[0].set_xticklabels([])
axes[0,0].text(.05,.85,"Fiducial", transform=axes[0,0].transAxes)
axes[0,1].text(.05,.85,"90Degree",transform=axes[0,1].transAxes)
axes[0,2].text(.05,.85,"TwoThirds",transform=axes[0,2].transAxes)
for i in range(len(snapshots)):
    axes[i,0].text(.6,.85,"t=%0.1f ns"%(snapshots[i]*1e9), transform=axes[i,0].transAxes)

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
basedir = "/global/project/projectdirs/m3761/PAPER2"
dirlist    = [["gray" ,"1D", basedir+"/Fiducial_1D"],
              ["black","2D", basedir+"/Fiducial_2D"],
              ["blue" ,"3D", basedir+"/Fiducial_3D"]]
for inputs in dirlist:
    filename = inputs[2]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[ind,0],val,inputs[0],'-',inputs[1],filename)

dirlist    = [["gray" ,"-" ,"1D", basedir+"/90Degree_1D"],
              ["black","-" ,"2D (out of plane)", basedir+"/90Degree_2D_outplane"],
              #["black","--","2D (in plane)", basedir+"/90Degree_2D_inplane"],
              ["blue","-","3D",basedir+"/90Degree_3D"]]
for inputs in dirlist:
    filename = inputs[3]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[ind,1],val,inputs[0],inputs[1],inputs[2],filename)

dirlist    = [["gray", "1D", basedir+"/TwoThirds_1D"],
              ["black","2D", basedir+"/TwoThirds_2D"],
              ["blue","3D", basedir+"/TwoThirds_3D/1"]]
for inputs in dirlist:
    filename = inputs[2]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[ind,2],val,inputs[0],'-',inputs[1],filename)

#Create custom legend
axes[0,0].legend(frameon=False,fontsize=13, loc=(.6,.45))
#axes[1,0].legend(frameon=False,fontsize=13, loc=(.45,.45))

############
# save pdf #
############
plt.savefig("power_spectrum.pdf", bbox_inches="tight")
