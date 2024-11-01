#!/usr/bin/env python3

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import yt
import numpy as np
import matplotlib.pyplot as plt
from yt import derived_field
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.render_source import VolumeSource
from yt.units import cm
from emu_yt_module import EmuDataset
import argparse
import textwrap

from mpl_toolkits.axes_grid1 import AxesGrid

#3 flavor neutrino derived fields: Number Density
#normalize by the trace
@derived_field(name="Norm", units="dimensionless", sampling_type="cell",force_override=True)
def _Norm(field, data):
    return np.abs(data["ee01"]) + np.abs(data["em01"])

#individual flavor pairing magnitudes, 0=electron, 1=mu, 2=tau
@derived_field(name="N01_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N01_Mag(field, data):
    return data["er01"]/data["Norm"]

#Diagonal components normalized
@derived_field(name="N00_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N00_Mag(field, data):
    return data["ee01"]/data["Norm"]
@derived_field(name="N11_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _N11_Mag(field, data):
    return data["em01"]/data["Norm"]

#total off-diagonal magnitude
@derived_field(name="OffDiag_Mag", units="dimensionless", sampling_type="cell",force_override=True)
def _Diag_Mag(field, data):
    return data["er01"]/data["Norm"]

#off-diagonal phases in degrees for each off-diagonal component is the arctan(Im/Re)
@derived_field(name="N01_Phase", units="dimensionless", display_name=r'\phi_{ex}\,\,(degrees)', sampling_type="cell",force_override=True)
def _N01_Phase(field, data):
    return data["ep01"]*(180/np.pi)

def get_output_dir(args):
    # construct the path to the output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.getcwd()
    return output_dir

def get_output_file(args, suffix):
    # construct a filename to write
    file = os.path.join(get_output_dir(args), f"{os.path.basename(args.plotfile)}_{suffix}")
    return file

def get_3D_selection(args, emu_ND):
    # by default, select the entire domain
    left_edge = emu_ND.ds.domain_left_edge.d
    right_edge = emu_ND.ds.domain_right_edge.d
    dim = emu_ND.get_num_dimensions()

    # check if the user selected edges are already the same as the domain edges
    # and if so, skip selecting a rectangular region for efficiency
    if (dim == 3 and
        np.max(np.abs(np.array(args.lo_edge) - left_edge)) <= np.finfo(float).eps and
        np.max(np.abs(np.array(args.hi_edge) - right_edge)) <= np.finfo(float).eps):
        # the dataset already matches our selection
        return emu_ND

    # the emu yt interface assumes 1D datasets are extended along z
    # and 2D datasets are extended along y, z
    extended_dims = [0,1,2][-dim:]

    # apply any user-supplied lo, hi edge selections
    if args.lo_edge:
        for i in extended_dims:
            left_edge[i] = args.lo_edge[i]

    if args.hi_edge:
        for i in extended_dims:
            right_edge[i] = args.hi_edge[i]

    # get the selected region in the dataset
    emu_ND_selection = emu_ND.get_rectangle(left_edge, right_edge)
    
    # convert the selection to a 3D dataset
    if emu_ND_selection.get_num_dimensions() < 3:
        return emu_ND_selection.to_3D()
    else:
        return emu_ND_selection

def do_phase_volume_render(args, emu_3D, field):
    # volume render a phase field in the dataset

    #initializing the scene to be rendered
    sc = yt.create_scene(emu_3D.ds, field)
    sc.annotate_axes(alpha=0.8)
    sc.annotate_domain(emu_3D.ds,color=[1, 1, 1, 0.85])

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
    sc.camera.focus = emu_3D.ds.domain_center
    sc.camera.resolution = 1024
    sc.camera.north_vector = [0, 0, 1]
    sc.camera.zoom(0.9)

    #format the timestamp
    #text_string = "t = {:.4f} ns".format(float(emu_3D.ds.current_time.to('ns')))

    ##save with transfer function also displayed on rendering image
    #sc.save_annotated(get_output_file(args, f"{field}_rendering.png"), sigma_clip=6,
    #                  text_annotate=[[(.1, 0.95), text_string, dict(fontsize="20")]],
    #                  label_fmt='%d')

    return sc

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
        example:
            python3 emu_phase_volume_render.py plt00000 -lo 0 0 0 -hi 8 8 8 -f all -o $(pwd)/volume_rendering

            [plt00000]: the name of a plotfile in the current working directory
            [-lo 0 0 0]: low edge of domain to volume render (units of cm)
            [-hi 8 8 8]: high edge of domain to volume render (units of cm)
            [-f all]: volume render all Phase fields
            [-o $(pwd)/volume_rendering]: save the output images in a local volume_rendering directory (must exist)
        '''))
    parser.add_argument("plotfile", type=str,
                        help="Name of the plotfile to process.")
    parser.add_argument("-lo", "--lo_edge", type=float, nargs=3,
                        help="Low edge of the 3D domain to volume-render in centimeters (default: use full domain)")
    parser.add_argument("-hi", "--hi_edge", type=float, nargs=3,
                        help="High edge of the 3D domain to volume-render in centimeters (default: use full domain)")
    parser.add_argument("-f", "--fields", type=str, nargs="+", default=["all"],
                        help="List of phase field names to volume-render. (default: all phase fields).")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="Absolute path to existing directory where to save the output files (default: use the current working directory)")
    args = parser.parse_args()

    # configure matplotlib
    rc = {"font.family" : "serif",'font.size':15}
    plt.rcParams.update(rc)

    # make a 3D dataset, if dimensionality is < 3
    # if dataset is already 3D, then simply select
    # the rectangle defined by [lo_edge, hi_edge]
    emu_ND = EmuDataset(args.plotfile)
    emu_3D = get_3D_selection(args, emu_ND)

    # do the volume rendering for all the phase fields in this plotfile
    if args.fields == ['all']:
        fields = [f for _, f in emu_3D.ds.derived_field_list if "Phase" in f]
    else:
        fields = args.fields

    for field in fields:
        sc = do_phase_volume_render(args, emu_3D, field)


    #new code for side-by-side plots:
    fig = plt.figure()

    # See http://matplotlib.org/mpl_toolkits/axes_grid/api/axes_grid_api.html
    # These choices of keyword arguments produce a four panel plot that includes
    # four narrow colorbars, one for each plot.  Axes labels are only drawn on the
    # bottom left hand plot to avoid repeating information and make the plot less
    # cluttered.
    grid = AxesGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        axes_pad=1.0,
        label_mode="1",
        share_all=False,
        cbar_location="right",
        cbar_mode="each",
        cbar_size="3%",
        cbar_pad="0%",
    )

    #sc.zoom(2)

    # For each plotted field, force the SlicePlot to redraw itself onto the AxesGrid
    # axes.
    plot = sc
    plot.figure = fig
    plot.axes = grid[1].axes
    plot.cax = grid.cbar_axes[1]

    # Finally, redraw the plot on the AxesGrid axes.
    sc.render()

    ax = fig.add_subplot(121)
    ax.plot([0, 1], [0,1])



    plt.savefig("multiplot_1x2.png")
