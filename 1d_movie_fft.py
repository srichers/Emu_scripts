import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FixedLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
from multiprocessing import Pool
import os
import imageio
import glob
import argparse
from emu_yt_module import FourierData

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str, help="Path to the file containing fft data for a simulation timeseries.")
parser.add_argument("-s", "--save_frames", action="store_true", help="Save each frame used for the movie.")
parser.add_argument("-o", "--output_name", type=str, default="1d_fft_matrix", help="Prefix of the output movie file.")
parser.add_argument("-fps", "--frames_per_second", type=int, default=10, help="Number of frames per second for the movie (default: 10).")
parser.add_argument("-p", "--processes", type=int, default=1, help="Number of processes to use.")
args = parser.parse_args()

# function for making one frame
def make_frame(fft, output_name, save_frames):
    # matplotlib configuration
    mpl.rcParams.update({'figure.autolayout': True})

    # dataset = yt dataset
    time = fft["time"]

    plt.clf()

    fig, axes = plt.subplots(3,3, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.05,wspace=0.05)

    # plot magnitudes in the upper half of the matrix
    axes[0,0].plot(kz, fft["N00_FFT"], linewidth=2)
    axes[0,1].plot(kz, fft["N01_FFT"], linewidth=2)
    axes[0,2].plot(kz, fft["N02_FFT"], linewidth=2)
    axes[1,1].plot(kz, fft["N11_FFT"], linewidth=2)
    axes[1,2].plot(kz, fft["N12_FFT"], linewidth=2)
    axes[2,2].plot(kz, fft["N22_FFT"], linewidth=2)

    # plot phases in the lower half of the matrix
    axes[1,0].plot(kz, fft["N01_FFT_phase"], linewidth=2)
    axes[2,0].plot(kz, fft["N02_FFT_phase"], linewidth=2)
    axes[2,1].plot(kz, fft["N12_FFT_phase"], linewidth=2)

    # set up image buffer and return figure and axes along with it
    fig.canvas.draw()

    if save_frames:
        plt.savefig("{}_{:05}.pdf".format(output_name, fft["index"]),bbox_inches="tight")

    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return (fig, axes, image)

def get_plotfile_frame(data):
    fig, axes, frame = make_frame(data, args.output_name, args.save_frames)
    plt.close(fig)
    return (data["time"], frame)

if __name__ == "__main__":
    # get the FFT data from the data file
    f = h5py.File(args.data,"r")

    t  = np.array(f["t"])       
    kz = np.array(f["kz"])*2.*np.pi      

    # get the FFT data for each time entry
    fft_timeseries = []

    # first get all the FFT data as a dictionary
    all_data = {}
    for key in f.keys():
        if not key in ["t", "kx", "ky", "kz"]:
            all_data[key] = np.array(f[key])

    # now separate it into a list of dictionaries,
    # 1 dictionary per time snapshot
    for i, ti in enumerate(t):
        now_data = {}
        now_data["index"] = i
        now_data["time"] = ti
        now_data["kz"] = kz
        for key in all_data.keys():
            now_data[key] = all_data[key][i,:]
        fft_timeseries.append(now_data)

    # plotfile frames stored as (plotfile, frame)
    plt_frames = []

    # get frames from plotfiles
    p = Pool(processes=args.processes)
    plt_frames = p.map(get_plotfile_frame, fft_timeseries)
    p.close()

    # frame buffer for movie
    plt_frames = sorted(plt_frames, key=lambda x: x[0])
    frames = [f for _, f in plt_frames if not f is None]

    # make movie
    imageio.mimsave("{}.gif".format(args.output_name), frames, fps=args.frames_per_second)