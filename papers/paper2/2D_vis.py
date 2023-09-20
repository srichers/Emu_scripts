import sys

import yt
import numpy as np
from yt import derived_field
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as tickeru
from matplotlib import colors
import glob

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

directory_in = "/ocean/projects/phy200048p/shared/2D/90deg_outofplane/"
directory_out = "/ocean/projects/phy200048p/shared/plots/90deg_outofplane/"
filenames = sorted(glob.glob(directory_in+"plt*"))
for filename in filenames:
    ds = yt.load(filename)
    sl = yt.SlicePlot(ds, 'x', 'OffDiag_Mag', origin='native')
    sl.set_zlim('OffDiag_Mag', 1e-6,4e-1)
    sl.set_log('OffDiag_Mag', False)
    sl.annotate_timestamp(corner='upper_left', time_format='t = {time:.4f} {units}', time_unit='ns', draw_inset_box=True)
    sl.save(directory_out+"{0}_{1}_yz.png".format(ds, 'Off_Diag_Mag'))
