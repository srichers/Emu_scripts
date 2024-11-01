import numpy as np
import h5py
#import scipy.optimize
import argparse

#############
# CONSTANTS #
#############

G = 6.67259e-8 # cm^3 g^-1 s^-2
c = 2.99792458e10 # cm/s
Msun = 1.99e33 # g
me = 9.1093837e-28 # g
erg_in_MeV = 1.60218e-6 # erg
m_neutron = 1.674920e-24 # g
cm_geometric = Msun*G/c**2
erg_geometric = Msun*c**2
MeV_geometric = erg_geometric / erg_in_MeV
edens_geom_erg_cc = erg_geometric / cm_geometric**3
edens_geom_MeV_cc = MeV_geometric / cm_geometric**3
ndens_geom_cc = 10.0e+50/cm_geometric**3

rho_geom = Msun / cm_geometric**3

hbar = 1.05457266e-27 # erg s
eV = 1.60218e-12 # erg
MeV = 1e6 * eV
GeV = 1e9 * eV
GF = 1.1663787e-5 / GeV**2 * (hbar*c)**3 # erg cm^3
sigma0 = 4*GF**2 * me**2 / (np.pi * hbar**4) # cm^2
g_A = -1.23
delta_np = 1.29322 * MeV


parser = argparse.ArgumentParser(description='Reads .dat files and writes to .hdf5 file')
parser.add_argument('-i', '--infile', dest='inf', type=str, help='input .dat file', metavar='', default='reduced0D.dat')
parser.add_argument('-o', '--output', dest='out', type=str, help='output .hdf5 file', metavar='', default='reduced0D_selection.h5')

args = parser.parse_args()

infilename = args.inf
print("Input filename: ", infilename)
print("\n")
infile = open(infilename, 'r')
infile_lines = infile.readlines()
infile.close()
#strip off header row:
infile_lines = infile_lines[1:]

t_floats = np.empty([len(infile_lines)])

N_avg_mag = np.empty([len(infile_lines), 2, 2])
Nbar_avg_mag = np.empty([len(infile_lines), 2, 2])
F_avg_mag = np.empty([len(infile_lines), 3, 2, 2])
Fbar_avg_mag = np.empty([len(infile_lines), 3, 2, 2])

t_ind = 1

#for beam_test/Evan_beam_series/neebar_*:
#N00_ind = 4
#N11_ind = 24
#N01_ind = 44
#
#Nbar00_ind = 5
#Nbar11_ind = 25
#
#Fx00_ind = 6
#Fy00_ind = 7
#Fz00_ind = 8
#Fx11_ind = 26
#Fy11_ind = 27
#Fz11_ind = 28
#
#Fbarx00_ind = 9
#Fbary00_ind = 10
#Fbarz00_ind = 11
#Fbarx11_ind = 29
#Fbary11_ind = 30
#Fbarz11_ind = 31

#for NSM_2.5/1res:
N00_ind = 4
N11_ind = 12
N01_ind = 20

Nbar00_ind = 5
Nbar11_ind = 13

Fx00_ind = 6
Fy00_ind = 7
Fz00_ind = 8
Fx11_ind = 14
Fy11_ind = 15
Fz11_ind = 16

Fbarx00_ind = 9
Fbary00_ind = 10
Fbarz00_ind = 11
Fbarx11_ind = 17
Fbary11_ind = 18
Fbarz11_ind = 19

for i, line in enumerate(infile_lines):
    t_floats[i]  = float(line.split()[t_ind].strip())

    N_avg_mag[i,0,0] = float(line.split()[N00_ind].strip())
    N_avg_mag[i,1,1] = float(line.split()[N11_ind].strip())
    N_avg_mag[i,0,1] = float(line.split()[N01_ind].strip())
    N_avg_mag[i,1,0] = N_avg_mag[i,0,1]

    Nbar_avg_mag[i,0,0] = float(line.split()[Nbar00_ind].strip())
    Nbar_avg_mag[i,1,1] = float(line.split()[Nbar11_ind].strip())
    Nbar_avg_mag[i,0,1] = None
    Nbar_avg_mag[i,1,0] = None

    F_avg_mag[i,0,0,0] = float(line.split()[Fx00_ind].strip())
    F_avg_mag[i,1,0,0] = float(line.split()[Fy00_ind].strip())
    F_avg_mag[i,2,0,0] = float(line.split()[Fz00_ind].strip())

    F_avg_mag[i,0,1,1] = float(line.split()[Fx11_ind].strip())
    F_avg_mag[i,1,1,1] = float(line.split()[Fy11_ind].strip())
    F_avg_mag[i,2,1,1] = float(line.split()[Fz11_ind].strip())

    F_avg_mag[i,:,0,1] = None
    F_avg_mag[i,:,1,0] = None

    Fbar_avg_mag[i,0,0,0] = float(line.split()[Fbarx00_ind].strip())
    Fbar_avg_mag[i,1,0,0] = float(line.split()[Fbary00_ind].strip())
    Fbar_avg_mag[i,2,0,0] = float(line.split()[Fbarz00_ind].strip())

    Fbar_avg_mag[i,0,1,1] = float(line.split()[Fbarx11_ind].strip())
    Fbar_avg_mag[i,1,1,1] = float(line.split()[Fbary11_ind].strip())
    Fbar_avg_mag[i,2,1,1] = float(line.split()[Fbarz11_ind].strip())

    Fbar_avg_mag[i,:,0,1] = None
    Fbar_avg_mag[i,:,1,0] = None

outfilename = args.out
print("Output filename: ", outfilename)
print("\n")
outfile = h5py.File(outfilename,"w")


outfile["t"] = t_floats
outfile["N_avg_mag"] = N_avg_mag
outfile["Nbar_avg_mag"] = Nbar_avg_mag
outfile["F_avg_mag"] = F_avg_mag
outfile["Fbar_avg_mag"] = Fbar_avg_mag

outfile.close()
