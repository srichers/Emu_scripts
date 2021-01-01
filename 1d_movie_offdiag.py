import importlib
plt_1d_offdiag_mag = importlib.import_module("1d_three_offdiag_mag")
plt_1d_offdiag_phase = importlib.import_module("1d_three_offdiag_phase")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FixedLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from multiprocessing import Pool
import os
import yt
import imageio
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--basedir", type=str, default="/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial", help="Path to the directory containing the plotfiles.")
parser.add_argument("-s", "--save_frames", action="store_true", help="Save each frame used for the movie.")
parser.add_argument("-o", "--output_name", type=str, default="1d_offdiag", help="Name of the output movie file.")
parser.add_argument("-f", "--frames_per_second", type=int, default=10, help="Number of frames per second for the movie (default: 10).")
parser.add_argument("-mt", "--max_time", type=float, help="Maximum simulation time for which to make the movie.")
parser.add_argument("-mz", "--max_zcoord", type=float, help="Maximum value of the z-coordinate to plot.")
parser.add_argument("-p", "--processes", type=int, default=1, help="Number of processes to use.")
args = parser.parse_args()

# function for making one frame
def make_frame(dataset, output_name, save_frames):
    # matplotlib configuration
    mpl.rcParams.update({'figure.autolayout': True})

    # dataset = yt dataset
    time = dataset.current_time
    ad = dataset.all_data()

    plt.clf()

    scale_fig_x_size = 6.0
    scale_fig_x_coord = 16.0
    ds_max_x = dataset.domain_right_edge[2]
    if args.max_zcoord and args.max_zcoord < ds_max_x:
        ds_max_x = args.max_zcoord
    fig_x_size = (ds_max_x / scale_fig_x_coord) * scale_fig_x_size
    fig, axes = plt.subplots(4,1, figsize=(fig_x_size,16))

    plt.subplots_adjust(hspace=0,wspace=0.05)
    plt_1d_offdiag_mag.snapshot_plot(ad, axes[:2], time)
    plt_1d_offdiag_phase.snapshot_plot(ad, axes[2:], time)

    xmax = ds_max_x
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(0,xmax)

    # configure magnitude plots
    axes[0].text(1.0,0.64, "t=%0.2f ns"%(time*1e9), ha="center", va="center")
    axes[0].legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc="lower left", ncol=6, mode="expand", borderaxespad=0., frameon=False)

    ylabel = r"$|n_{ab}| /\mathrm{Tr}(n_{ab})$"
    axes[0].set_ylabel(ylabel)    
    axes[0].set_ylim(0,.5)
    axes[0].set_xticklabels([])
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())

    ylabel = r"$10^6\times |\mathbf{f}^{(x)}_{ab}| /\mathrm{Tr}(n_{ab})$"
    axes[1].set_ylabel(ylabel)    
    axes[1].set_ylim(0,3.99)
    axes[1].set_xticklabels([])
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())

    # configure phase plots
    alpha = 0.9

    ylabel = r"$\phi_{n_{ab}}$"
    axes[2].set_ylabel(ylabel)    
    axes[2].set_ylim(-1,1)
    axes[2].yaxis.set_minor_locator(MultipleLocator(.25))
    axes[2].yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
    axes[2].set_yticklabels([r"$-\pi$",0,r"$\pi$"])
    axes[2].set_xticklabels([])

    ylabel = r"$\phi_{\mathbf{f}^{(x)}_{ab}}$"
    axes[3].set_ylabel(ylabel)    
    axes[3].set_ylim(-1,1)
    axes[3].yaxis.set_minor_locator(MultipleLocator(.25))
    axes[3].yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
    axes[3].set_yticklabels([r"$-\pi$",0,r"$\pi$"])
    axes[3].set_xlabel(r"$z\,(\mathrm{cm})$")

    # final adjustments to the figure
    fig.align_xlabels(axes)
    fig.align_ylabels(axes)

    # set up image buffer and return figure and axes along with it
    fig.canvas.draw()

    if save_frames:
        plt.savefig("{}_{}.pdf".format(output_name, pltname),bbox_inches="tight")

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return (fig, axes, image)

def get_plotfile_frame(plotfile):
    pltname = os.path.basename(plotfile)
    ds = yt.load(plotfile)

    if args.max_time and ds.current_time > args.max_time:
        return (pltname, None)
    else:
        fig, axes, frame = make_frame(ds, args.output_name, args.save_frames)
        plt.close(fig)
        return (pltname, frame)

if __name__ == "__main__":
    # get the plotfiles in the directory specified
    plotfiles = sorted(glob.glob(os.path.join(args.basedir, "plt*")))

    # plotfile frames stored as (plotfile, frame)
    plt_frames = []

    # get frames from plotfiles
    p = Pool(processes=args.processes)
    plt_frames = p.map(get_plotfile_frame, plotfiles)
    p.close()

    # frame buffer for movie
    plt_frames = sorted(plt_frames, key=lambda x: x[0])
    frames = [f for _, f in plt_frames if not f is None]

    # make movie
    imageio.mimsave("{}.gif".format(args.output_name), frames, fps=args.frames_per_second)
