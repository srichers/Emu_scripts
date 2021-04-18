import sys
import os
import yt
import numpy as np
from yt import derived_field
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colorbar
import glob

print("imported all packages!")

slice_direction = 'z'

#3 flavor neutrino derived fields: Number Density
#normalize by the trace
@derived_field(name="Norm", units="dimensionless", sampling_type="cell",force_override=True)
def _Norm(field, data):
    return np.abs(data["N00_Re"]) + np.abs(data["N11_Re"]) + np.abs(data["N22_Re"])
@derived_field(name="N01_Norm", units="dimensionless", sampling_type="cell",force_override=True)
def _N01_Norm(field, data):
    return np.abs(data["N00_Re"]) + np.abs(data["N11_Re"])
@derived_field(name="N02_Norm", units="dimensionless", sampling_type="cell",force_override=True)
def _N02_Norm(field, data):
    return np.abs(data["N00_Re"]) + np.abs(data["N22_Re"])
@derived_field(name="N12_Norm", units="dimensionless", sampling_type="cell",force_override=True)
def _N12_Norm(field, data):
    return np.abs(data["N11_Re"]) + np.abs(data["N22_Re"])

#individual flavor pairing magnitudes, 0=electron, 1=mu, 2=tau
@derived_field(name="N01_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N01_Mag(field, data):
    return np.sqrt((data["N01_Im"]/data["N01_Norm"])**2 + (data["N01_Re"]/data["N01_Norm"])**2)
@derived_field(name="N02_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N02_Mag(field, data):
    return np.sqrt((data["N02_Im"]/data["N02_Norm"])**2 + (data["N02_Re"]/data["N02_Norm"])**2)
@derived_field(name="N12_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N12_Mag(field, data):
    return np.sqrt((data["N12_Im"]/data["N12_Norm"])**2 + (data["N12_Re"]/data["N12_Norm"])**2)

#Diagonal components normalized
@derived_field(name="N00_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N00_Mag(field, data):
    return data["N00_Re"]/data["Norm"]
@derived_field(name="N11_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N11_Mag(field, data):
    return data["N11_Re"]/data["Norm"]
@derived_field(name="N22_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N22_Mag(field, data):
    return data["N22_Re"]/data["Norm"]

#total off-diagonal magnitude
@derived_field(name="OffDiag_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _Diag_Mag(field, data):
    return np.sqrt((data["N01_Im"]/data["Norm"])**2 + (data["N01_Re"]/data["Norm"])**2 + (data["N02_Im"]/data["Norm"])**2 + (data["N02_Re"]/data["Norm"])**2 + (data["N12_Im"]/data["Norm"])**2 + (data["N12_Re"]/data["Norm"])**2)

#off-diagonal phases in degrees for each off-diagonal component is the arctan(Im/Re)
@derived_field(name="N01_Phase", units="dimensionless", sampling_type="cell",force_override=True)
def _N01_Phase(field, data):
    return np.arctan2(data["N01_Im"],data["N01_Re"])*(180/np.pi)
@derived_field(name="N02_Phase", units="dimensionless", sampling_type="cell",force_override=True)
def _N02_Phase(field, data):
    return np.arctan2(data["N02_Im"],data["N02_Re"])*(180/np.pi)
@derived_field(name="N12_Phase", units="dimensionless", sampling_type="cell",force_override=True)
def _N12_Phase(field, data):
    return np.arctan2(data["N12_Im"],data["N12_Re"])*(180/np.pi)

print("defined all fields!")

plt.rc('axes', labelsize=18, linewidth=2)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams["xtick.major.size"] = 6
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["ytick.major.size"] = 6
plt.rcParams["ytick.minor.size"] = 3

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

yt.funcs.mylog.setLevel(100)

filenames = sorted(glob.glob("plt*"))
for filename in filenames:
    print(filename)
    ds = yt.load(filename)
    slc = ds.slice(slice_direction, 0.)
    #use fixed resolution buffer to extract 2D slice data object information
    if slice_direction == 'x':
        i1=1
        i2=2
    if slice_direction == 'y':
        i1 = 0
        i2 = 2
    if slice_direction == 'z':
        i1 = 0
        i2 = 1
    frb = slc.to_frb(width = (ds.domain_right_edge[i1].value - ds.domain_left_edge[i1].value, 'cm'),
                     height = (ds.domain_right_edge[i2].value - ds.domain_left_edge[i2].value, 'cm'),
                     resolution = (ds.domain_dimensions[i1], ds.domain_dimensions[i2]))
    #list of fields to plot ordered by row and then column
    fields = [['N00_Mag','N01_Mag','N02_Mag'],
              ['N01_Phase','N11_Mag','N12_Mag'],
              ['N02_Phase','N12_Phase','N22_Mag']]
    labels = [[r'$N_{ee}$',r'$N_{e\mu}$',r'$N_{e\tau}$'],
              [r'$\phi_{e\mu}$',r'$N_{\mu\mu}$',r'$N_{\mu\tau}$'],
              [r'$\phi_{e\tau}$',r'$\phi_{\mu\tau}$',r'$N_{\tau\tau}$']]
    
    # Set up a 3x3 figure
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15), sharex=True, sharey=True)
    print("created figure")
    #defining the two colorbars to be used
    cax1 = fig.add_axes([.93, 0.122, 0.03, 0.76])
    norm1 = colors.Normalize(vmin=0, vmax=0.6)
    cmap = plt.get_cmap('viridis')
    cb1 = colorbar.ColorbarBase(cax1, cmap=cmap,norm=norm1,orientation='vertical')
    cb1.set_label(r'$N_{ij} / \mathrm{Tr}(N)$', fontsize=22)
    cb1.ax.tick_params(axis='y', which='both', direction='in')
    cb1.minorticks_on()
    cax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    cax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    
    cax2 = fig.add_axes([0.04, 0.122, 0.03, 0.76])
    norm2 = colors.Normalize(vmin=-180, vmax=180)
    cmap2 = plt.get_cmap('twilight')
    cb2 = colorbar.ColorbarBase(cax2, cmap=cmap2,norm=norm2,orientation='vertical')
    cax2.yaxis.set_ticks_position('left')
    cax2.yaxis.set_label_position('left')
    cb2.set_label(r'$\phi_{ij}$ (degrees)', fontsize=22)
    cb2.ax.tick_params(axis='y', which='both', direction='in')
    cax2.minorticks_on()
    cax2.yaxis.set_major_locator(ticker.MultipleLocator(45))
    cax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    
    for i in range(0,3):
        for j in range(0,3):
            ax = axes[i,j]
            ax.minorticks_on()
            ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(2))
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(2))
            ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
            ax.text(0.1,0.9, "{}".format(labels[i][j]), fontsize=18, ha="center", va="center", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.75, edgecolor='.75'))
            #setting the phase plots
            if ax == axes[1,0] or ax == axes[2,0] or ax == axes[2,1]:
                im = ax.imshow(frb[fields[i][j]].d,origin='lower',extent=[ds.domain_left_edge[1].value,ds.domain_right_edge[1].value,ds.domain_left_edge[2].value,ds.domain_right_edge[2].value], vmin=-180, vmax=180, cmap='twilight')
                #setting colorbar limits
                im.set_clim(-180,180)
                #setting the N magnitudes plots
            else:
                #origin=lower keeps image from getting flipped, extent gives the grid coordinates in physical units
                im = ax.imshow(frb[fields[i][j]].d,origin='lower',extent=[ds.domain_left_edge[1].value,ds.domain_right_edge[1].value,ds.domain_left_edge[2].value,ds.domain_right_edge[2].value], vmin=0., vmax=0.6, cmap='viridis')
                #setting colorbar limits
                im.set_clim(0.,0.6)
    label_arr = ["x","y","z"]
    for a in axes[-1,:]: a.set_xlabel(label_arr[i1]+' (cm)')
    for a in axes[:,0]: a.set_ylabel(label_arr[i2]+' (cm)')

    #plt.tick_params(axis='both', which='both', direction='in', right=True,top=True)
    #increases the spacing between subplots so axis labels don't overlap
    plt.subplots_adjust(hspace=0.05,wspace=0.)
    #fig.tight_layout()
    #plt.show()
    #fig.savefig(directory_out+"/"+"{0}_3x3subplot.pdf".format(ds),dpi=300, bbox_inches="tight")
    fig.savefig("{0}_3x3subplot.png".format(ds),dpi=300, bbox_inches="tight")
    print("successfully saved figure")
    plt.close()
