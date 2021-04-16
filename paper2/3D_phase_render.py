import yt
import numpy as np
import matplotlib.pyplot as plt
from yt import derived_field
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.render_source import VolumeSource
from yt.units import cm

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
@derived_field(name="N01_Phase", units="dimensionless", display_name='\phi_{e\mu}\,\,(degrees)', sampling_type="cell",force_override=True)
def _N01_Phase(field, data):
    return np.arctan2(data["N01_Im"],data["N01_Re"])*(180/np.pi)
@derived_field(name="N02_Phase", units="dimensionless", display_name='\phi_{e\tau}\,\,(degrees)', sampling_type="cell",force_override=True)
def _N02_Phase(field, data):
    return np.arctan2(data["N02_Im"],data["N02_Re"])*(180/np.pi)
@derived_field(name="N12_Phase", units="dimensionless", display_name='\phi_{\mu\tau}\,\,(degrees)', sampling_type="cell",force_override=True)
def _N12_Phase(field, data):
    return np.arctan2(data["N12_Im"],data["N12_Re"])*(180/np.pi)

rc = {"font.family" : "serif",'font.size':15}
plt.rcParams.update(rc)

#alternate volume renderings with color determined by the off-diagonal phase
base_dir = "/global/project/projectdirs/m3761/3D/128r64d_128n800s16mpi/"
#five representative snapshots
files = ["run1/plt00100","run1/plt00250","run1/plt00400","run1/plt00800","run3/plt02200"]

for f in files:
    ds = yt.load(base_dir+f)
    field='N01_Phase'
    #initializing the scene to be rendered
    sc = yt.create_scene(ds, field)
    sc.annotate_axes(alpha=0.8)
    sc.annotate_domain(ds,color=[1, 1, 1, 0.85])

    #setting up the color transfer function
    source = sc[0]
    source.set_field(field)
    v_min = -180
    v_max = 180
    source.tfh.set_log(False) #Apparently need to set_log to false before doing anything else
    source.set_log(False) #apparently you also need this command to actually turn off the log setting!
    source.tfh.set_bounds([v_min,v_max])
    source.tfh.grey_opacity = True
    tf = yt.ColorTransferFunction((v_min,v_max))
    
    #Manually adding gaussians to the transfer function, grabbing colors from twilight colormap
    layers = [-180, -90, 0, 90, 180]
    for l in layers:
        tf.sample_colormap(l, 0.2*(v_max - v_min)/5, alpha = 50, colormap = 'twilight_shifted')
    
    #If you wanted to add several evenly spaced gaussians, use this instead:
    #L = 5 #number of layers to add
    #tf.add_layers(L, w=0.2*(v_max - v_min)/L, alpha=[50,50,50,50,50], colormap = 'twilight_shifted')
    
    source.tfh.tf = tf

    #camera parameters
    sc.camera.focus = ds.domain_center
    sc.camera.resolution = 1024
    sc.camera.north_vector = [0, 0, 1]
    sc.camera.zoom(0.9)

    #plot the transfer function
    source.tfh.plot(base_dir+'{0}_{1}_transfer_function.png'.format(ds,field))

    #format the timestamp
    text_string = "t = {:.4f} ns".format(float(ds.current_time.to('ns')))

    #save with transfer function also displayed on rendering image
    sc.save_annotated(base_dir+'{0}_{1}_rendering.png'.format(ds,field), sigma_clip=6, text_annotate=[[(.1, 0.95), text_string, dict(fontsize="20")]], label_fmt='%d')
