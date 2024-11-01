#FLASH data is stored under the following convention: (0,1,2) = (z,y,x)
#This is reflected in all of the below arrays, except blk_coords

import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
#import plotly.graph_objects as go
#import plotly.subplots
import os
import argparse


def get_time(dset):
    for thing in dset:
        if thing[0].strip()==b'time':
            return thing[1]


def get_cell_coordinates(h5file):

    cpb = np.empty([3], dtype=int)
    dset = h5file["/integer scalars"]
    for thing in dset:
        if thing[0].strip()==b'nzb':
            cpb[0] = int(thing[1])
        elif thing[0].strip()==b'nyb':
            cpb[1] = int(thing[1])
        elif thing[0].strip()==b'nxb':
            cpb[2] = int(thing[1])

    nblock = np.empty([3], dtype=int)
    dset = h5file["/integer runtime parameters"]
    for thing in dset:
        if thing[0].strip()==b'nblockz':
            nblock[0] = int(thing[1])
        elif thing[0].strip()==b'nblocky':
            nblock[1] = int(thing[1])
        elif thing[0].strip()==b'nblockx':
            nblock[2] = int(thing[1])

    domain_limits = np.empty([3,2])
    dset = h5file["/real runtime parameters"]
    for thing in dset:
        if thing[0].strip()==b'zmin':
            domain_limits[0,0] = float(thing[1])
        elif thing[0].strip()==b'zmax':
            domain_limits[0,1] = float(thing[1])
        elif thing[0].strip()==b'ymin':
            domain_limits[1,0] = float(thing[1])
        elif thing[0].strip()==b'ymax':
            domain_limits[1,1] = float(thing[1])
        elif thing[0].strip()==b'xmin':
            domain_limits[2,0] = float(thing[1])
        elif thing[0].strip()==b'xmax':
            domain_limits[2,1] = float(thing[1])

    cell_coords = [[], [], []]
    for i in range(3):
        #x serves as a generic label:
        x1 = domain_limits[i,0]
        x2 = domain_limits[i,1]
        num_cells = cpb[i]*nblock[i]
        for j in range(num_cells):
            xloc = x1 + (0.5 + float(j))*(x2 - x1)/float(num_cells)
            cell_coords[i].append(xloc)

    return cell_coords


def get_Nex_entire_grid(h5file, nu_str, cell_coords):

    if nu_str == "nu":
        ee_label = "ee01"
        xx_label = "em01"
        ex_label = "er01"
    else:
        ee_label = "ea01"
        xx_label = "en01"
        ex_label = "es01"

    dset_blks = h5file["/integer scalars"]
    for thing in dset_blks:
        if thing[0].strip()==b"globalnumblocks":
            blks = int(thing[1])

    dset_ee = h5file[ee_label]
    dset_xx = h5file[xx_label]
    dset_ex = h5file[ex_label]
    Nex_tr = np.empty([len(cell_coords[0]), len(cell_coords[1]), len(cell_coords[2])])
    blk_coords = np.array(h5file['bounding box'])[:,:,:]
    #rectangular grid:
    dwidth = np.array([abs(cell_coords[0][1] - cell_coords[0][0]), \
        abs(cell_coords[1][1] - cell_coords[1][0]), \
        abs(cell_coords[2][1] - cell_coords[2][0])])
    for blk in range(blks):
        N_trace = dset_ee[blk] + dset_xx[blk]
        Nex = dset_ex[blk]
        inds = get_inds_bbox(blk_coords[blk], dwidth)
        Nex_tr[inds[0,0]:inds[0,1], inds[1,0]:inds[1,1], inds[2,0]:inds[2,1]] = Nex/N_trace

    return Nex_tr


def get_phase_entire_grid(h5file, nu_str, cell_coords):

    if nu_str == "nu":
        phi_label = "ep01"
    else:
        phi_label = "eq01"

    dset_blks = h5file["/integer scalars"]
    for thing in dset_blks:
        if thing[0].strip()==b"globalnumblocks":
            blks = int(thing[1])

    dset_phi = h5file[phi_label]
    phi_array = np.empty([len(cell_coords[0]), len(cell_coords[1]), len(cell_coords[2])])
    blk_coords = np.array(h5file['bounding box'])[:,:,:]
    #rectangular grid:
    dwidth = np.array([abs(cell_coords[0][1] - cell_coords[0][0]), \
            abs(cell_coords[1][1] - cell_coords[1][0]), \
            abs(cell_coords[2][1] - cell_coords[2][0])])
    for blk in range(blks):
        phi = dset_phi[blk]
        inds = get_inds_bbox(blk_coords[blk], dwidth)
        phi_array[inds[0,0]:inds[0,1], inds[1,0]:inds[1,1], inds[2,0]:inds[2,1]] = phi

    return phi_array

def get_inds_bbox(coords, dwidth):
    inds = np.empty([3,2], dtype=int)
    for i in range(3):
        dom_low = coords[2-i,0]
        inds[i,0] = int(round(dom_low/dwidth[i]))
        dom_high = coords[2-i,1]
        inds[i,1] = int(round(dom_high/dwidth[i]))
    return inds

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


#FLASH quantities
e01_energy = 50.0 #MeV
MeV_to_codeenergy = 1.60217733e-6*5.59424238e-55 #code energy/MeV
cm_to_codelength = 6.77140812e-06 #code length/cm

parser = argparse.ArgumentParser(description='Reads hdf5 files and plots output in 2D space for given time')
parser.add_argument('-i', '--infile', dest='inf', type=str, help='input .hdf5 file', metavar='', default='sim_hdf5_chk_0000')
parser.add_argument('-o', '--outfile', dest='out', type=str, help="output file name", metavar='', default="[use naming convention]")
parser.add_argument('-n', '--nutype', dest='nu', type=str, help="neutrino (nu) or anti-neutrino (bnu)", metavar='', default="nu")
parser.add_argument('-p', '--pencildir', dest='pen', type=str, help='pencil direction (x,y,z)', metavar='', default='z')
parser.add_argument('-f', '--fixedpoints', dest='fix', type=str, help='string of 2 integers for the fixed indices of a particular pencil direction', metavar='', default='random')

args = parser.parse_args()

filename = args.inf
time_str = filename[-4:]
namestr = args.out

dim_str = args.pen
if dim_str == "x":
    dim_int = 2
    cdim1 = 1
    cdim1_str = "y"
    cdim2 = 0
    cdim2_str = "z"
    hlabel = r"$x\,{\rm (cm)}$"
    save_dims_str = "_x_y{}_z{}"
elif dim_str == "y":
    dim_int = 1
    cdim1 = 2
    cdim1_str = "x"
    cdim2 = 0
    cdim2_str = "z"
    hlabel = r"$y\,{\rm (cm)}$"
    save_dims_str = "_y_x{}_z{}"
else:
    dim_int = 0
    cdim1 = 2
    cdim1_str = "x"
    cdim2 = 1
    cdim2_str = "y"
    hlabel = r"$z\,{\rm (cm)}$"
    save_dims_str = "_z_x{}_y{}"

nu_str = args.nu
if nu_str != "nu":
    nu_str == "bnu"


f = h5py.File(filename,"r")
tau = get_time(f["/real scalars"])

cell_coords = get_cell_coordinates(f)

mod_data = get_Nex_entire_grid(f, nu_str, cell_coords)
phi_data = get_phase_entire_grid(f, nu_str, cell_coords)

f.close()


fixed_points_str = args.fix
fixed_points_str_array = fixed_points_str.split()

fp = np.empty([2], dtype=int)
if len(fixed_points_str_array) == 2:
    if RepresentsInt(fixed_points_str_array[0]) and RepresentsInt(fixed_points_str_array[1]):
        for i in range(2):
            fp[i] = int(fixed_points_str_array[i])
            if i == 0:
                odim = cdim1
            else:
                odim = cdim2
            last_cell_p1 = len(cell_coords[odim])
            if fp[i] < 0:
                fp[i] = last_cell_p1 + fp[i]
            if fp[i] < 0:
                fp[i] = 0
            if fp[i] > (fp[i] - 1):
                fp[i] = fp[i] - 1
    else:
        fp[0] = int(np.random.random()*float(len(cell_coords[cdim1])))
        fp[1] = int(np.random.random()*float(len(cell_coords[cdim2])))
else:
    fp[0] = int(np.random.random()*float(len(cell_coords[cdim1])))
    fp[1] = int(np.random.random()*float(len(cell_coords[cdim2])))


if dim_int == 0:
    mod_pencil = mod_data[:,fp[0],fp[1]]
    phi_pencil = phi_data[:,fp[0],fp[1]]
elif dim_int == 1:
    mod_pencil = mod_data[fp[0],:,fp[1]]
    phi_pencil = phi_data[fp[0],:,fp[1]]
else:
    mod_pencil = mod_data[fp[0],fp[1],:]
    phi_pencil = phi_data[fp[0],fp[1],:]


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


#######################
# LABELS FOR PLOTTING #
#######################

#print("Grid dimension:",np.shape(scalar_data))

print("modulus min/max", np.min(mod_pencil), np.max(mod_pencil))

fig = plt.figure()
ax = fig.gca()

##############
# formatting #
##############
ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.minorticks_on()
ax.set_xlabel(hlabel)
#ax.set_ylabel(r"${\rm Co}[N_{ex}]/{\rm Tr}[N]$")
ax.set_ylabel(r"${\rm mod}[N_{ex}]$")
#ax.set_ylabel(r"${\rm arg}[N_{ex}]$")
print("Time (ns): {:.2e}".format(1.e+9*tau))
#ax.set_title(r"$t={:.2E}\,{{\rm ns}};\,{}={:.2E}\,{{\rm cm}};\,{}={:.2E}\,{{\rm cm}}$".format( \
#        1.e+9*tau, cdim1_str, cell_coords[cdim1][fp[0]], cdim2_str, cell_coords[cdim2][fp[1]]), fontsize=16)


#real = ax.plot(cell_coords[dim_int], mod_pencil*np.cos(phi_pencil), "r-", label=r"${\rm Re}$")
#imag = ax.plot(cell_coords[dim_int], mod_pencil*np.sin(phi_pencil), "b--", label=r"${\rm Im}$")
mod = ax.plot(cell_coords[dim_int], mod_pencil, "b-", label=r"${\rm mod}$")
#phase = ax.plot(cell_coords[dim_int], phi_pencil, "r-", label=r"$\phi_{ex}$")

ax.legend(loc="best")
#ax.legend(loc="lower right")

if namestr == "[use naming convention]":
    pwd_str = os.getcwd()
    #namestr = pwd_str + "/" + "Nex_RI_" + nu_str + "_" + time_str + save_dims_str.format(fp[0], fp[1]) + ".pdf"
    #namestr = pwd_str + "/" + "Nex_mod_" + nu_str + "_" + time_str + save_dims_str.format(fp[0], fp[1]) + ".pdf"
    namestr = pwd_str + "/mod_v_z/" + "Nex_mod_" + nu_str + "_" + time_str + save_dims_str.format(fp[0], fp[1]) + ".pdf"
    #namestr = pwd_str + "/phi_v_z/" + "Nex_phi_" + nu_str + "_" + time_str + save_dims_str.format(fp[0], fp[1]) + ".pdf"

plt.savefig(namestr, bbox_inches="tight")
