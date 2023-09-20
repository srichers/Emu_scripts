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

filename = "plt01340"
field = "N01_Phase"

# list directories (fiducial)
directory_1d = "/ocean/projects/phy200048p/shared/1D/1D_fiducial"
directory_2d = "/ocean/projects/phy200048p/shared/2D/fiducial_2D"
directory_3d = "/ocean/projects/phy200048p/shared/3D/fiducial_3D/1"
slicedir_3d = "x"

# list directories (90 inplane)
#directory_1d = "/ocean/projects/phy200048p/shared/1D/90deg"
#directory_2d = "/ocean/projects/phy200048p/shared/2D/90deg_inplane"
#directory_3d = ""
#slicedir_3d = "x"

# list directories (90 outofplane)
#directory_1d = "/ocean/projects/phy200048p/shared/1D/90deg"
#directory_2d = "/ocean/projects/phy200048p/shared/2D/90deg_outofplane"
#directory_3d = ""
#slicedir_3d = "y"



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

plt.rcParams['font.size'] = 22
plt.rcParams['xtick.major.size'] = 7
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.minor.width'] = 2
plt.rcParams['ytick.major.size'] = 7
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.minor.width'] = 2
plt.rcParams['axes.linewidth'] = 2

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

yt.funcs.mylog.setLevel(100)

# Set up a 3x3 figure
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,7),
                         sharex=False, sharey=True,
                         gridspec_kw={'width_ratios':[1./7.,1,1]})

#defining the colorbar to be used
cax2 = fig.add_axes([.93, 0.13, 0.03, 0.73])
norm2 = colors.Normalize(vmin=-180, vmax=180)
cmap2 = plt.get_cmap('twilight')
cb2 = colorbar.ColorbarBase(cax2, cmap=cmap2,norm=norm2,orientation='vertical')
cb2.set_label(r'$\phi_{ij}$ (degrees)', fontsize=22)
cb2.ax.tick_params(axis='y', which='both', direction='in', right=True,left=False)
cax2.minorticks_on()
cax2.yaxis.set_major_locator(ticker.MultipleLocator(45))
cax2.yaxis.set_minor_locator(ticker.MultipleLocator(5))

for ax in axes:
    ax.minorticks_on()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)

#axes[0].set_xlabel('y (cm)')
axes[1].set_xlabel('y (cm)')
axes[2].set_xlabel('y (cm)')
axes[0].set_ylabel('z (cm)')
axes[0].set_xticklabels([])

###########
# 1D PLOT #
###########
ds = yt.load(directory_1d+'/'+filename)
print("1D:",ds.current_time)
slc = ds.slice('x', 0.)
#use fixed resolution buffer to extract 2D slice data object information
frb = slc.to_frb(
    width  = (ds.domain_right_edge[1].value - ds.domain_left_edge[1].value, 'cm'),
    height = (ds.domain_right_edge[2].value - ds.domain_left_edge[2].value, 'cm'),
    resolution = (ds.domain_dimensions[1], ds.domain_dimensions[2]))
im = axes[0].imshow(frb[field].d,origin='lower',vmin=-180, vmax=180, cmap='twilight',
                    extent=[ds.domain_left_edge[1].value,
                            ds.domain_right_edge[1].value,
                            ds.domain_left_edge[2].value,
                            ds.domain_right_edge[2].value],
                    aspect=.88)
#setting colorbar limits
im.set_clim(-180,180)
axes[0].text(.5,7.5,"1D",
             backgroundcolor="white", bbox=dict(alpha=0.5,edgecolor="white",facecolor="white"),
             horizontalalignment='center',verticalalignment='center')
            
###########
# 2D PLOT #
###########
ds = yt.load(directory_2d+'/'+filename)
print("2D:",ds.current_time)
slc = ds.slice('x', 0.)
#use fixed resolution buffer to extract 2D slice data object information
frb = slc.to_frb(
    width  = (ds.domain_right_edge[1].value - ds.domain_left_edge[1].value, 'cm'),
    height = (ds.domain_right_edge[2].value - ds.domain_left_edge[2].value, 'cm'),
    resolution = (ds.domain_dimensions[1], ds.domain_dimensions[2]))
im = axes[1].imshow(frb[field].d,origin='lower',vmin=-180, vmax=180, cmap='twilight',
                    extent=[ds.domain_left_edge[1].value,
                            ds.domain_right_edge[1].value,
                            ds.domain_left_edge[2].value,
                            ds.domain_right_edge[2].value])
#setting colorbar limits
im.set_clim(-180,180)
axes[1].text(4,7.5,"2D",
             backgroundcolor="white", bbox=dict(alpha=0.5,edgecolor="white",facecolor="white"),
             horizontalalignment='center',verticalalignment='center')
            
###########
# 3D PLOT #
###########
ds = yt.load(directory_3d+'/'+filename)
print("3D:",ds.current_time)
slc = ds.slice(slicedir_3d, 0.)
#use fixed resolution buffer to extract 2D slice data object information
frb = slc.to_frb(
    width  = (ds.domain_right_edge[1].value - ds.domain_left_edge[1].value, 'cm'),
    height = (ds.domain_right_edge[2].value - ds.domain_left_edge[2].value, 'cm'),
    resolution = (ds.domain_dimensions[1], ds.domain_dimensions[2]))
im = axes[2].imshow(frb[field].d,origin='lower',vmin=-180, vmax=180, cmap='twilight',
                    extent=[ds.domain_left_edge[1].value,
                            ds.domain_right_edge[1].value,
                            ds.domain_left_edge[2].value,
                            ds.domain_right_edge[2].value])
#setting colorbar limits
im.set_clim(-180,180)
axes[2].text(4,7.5,"3D",
             backgroundcolor="white", bbox=dict(alpha=0.5,edgecolor="white",facecolor="white"),
             horizontalalignment='center',verticalalignment='center')
            

#plt.tick_params(axis='both', which='both', direction='in', right=True,top=True)
#increases the spacing between subplots so axis labels don't overlap
plt.subplots_adjust(hspace=0.0,wspace=0.1)
fig.savefig("{0}_".format(ds)+field+"_dimension_phase_compare.png",dpi=300, bbox_inches="tight")
