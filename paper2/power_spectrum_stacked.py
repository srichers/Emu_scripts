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

fig, axes = plt.subplots(2,3, figsize=(16,6), sharex = True, sharey = True)
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

def manual_legend(lab,col,line,ax,pos):
    #create manual entries for legend, with 2 columns, one indicating simulation (color) and the other timestep (linestyle)
    lines=[]
    for r in range(len(lab)):
        if r < 3:
            lines.append(Line2D([0], [0], color=col[r], linestyle=line[0], lw=4))
        if r >= 3:
            lines.append(Line2D([0], [0], color=col[1], linestyle=line[r-3]))
    #print(lines)
    ax.legend(lines,lab,frameon=False,ncol=2,fontsize=12, loc=pos)

#############
# plot data #
#############
dirlist    = ["m3018/Emu/PAPER/1D/converge_direction/1D_dir64","m3018/Emu/2D/manydirections64","m3761/3D/128r64d_128n800s16mpi"]
colors     = ["gray"          ,"black"         ,"blue"          ]
labels     = ["1D","2D","3D"]
for i in range(len(dirlist)):
    filename = "/global/project/projectdirs/"+dirlist[i]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[0,ind],val,colors[i],'-',labels[i],filename)

dirlist    = ["m3018/Emu/PAPER/1D/fbar_direction/90","m3018/Emu/2D/90deg_inplane","m3018/Emu/2D/90deg_outofplane"]
linestyles = ["-"       ,"-"               ,"--"                 ]
colors     = ["gray"    ,"black"           ,"black"              ]
labels     = ["1D","2D (in plane)","2D (out of plane)"]
for i in range(len(dirlist)):
    filename = "/global/project/projectdirs/"+dirlist[i]+"/reduced_data_fft_power.h5"
    for ind,val in enumerate(snapshots):
        plotdata(variable,flavor,axes[1,ind],val,colors[i],linestyles[i],labels[i],filename)

#Create custom legend
axes[0,0].legend(frameon=False,fontsize=13, loc=(.6,.45))
axes[1,0].legend(frameon=False,fontsize=13, loc=(.45,.45))

############
# save pdf #
############
plt.savefig("power_spectrum.pdf", bbox_inches="tight")
