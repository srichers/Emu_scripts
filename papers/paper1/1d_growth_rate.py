# Calculate growth rates for a "reduced_data" file

import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datafile", type=str, help="Path to a reduced_data hdf5 file.")
parser.add_argument("-t1", "--time_start", type=float, default=0.15e-9, help="Starting time for the fit interval [seconds] (default: 0.15e-9)")
parser.add_argument("-t2", "--time_end", type=float, default=0.25e-9, help="Ending time for the fit interval [seconds] (default: 0.25e-9)")
parser.add_argument("-p", "--print", action="store_true", help="Print growth rates.")
parser.add_argument("-o", "--output", type=str, default="growth_rates", help="Name of the output file.")
parser.add_argument("-l", "--label", type=str, default="", help="Label to write into the output hdf5 file (default is '').")
args = parser.parse_args()

# Read averaged data
avgData = h5py.File(args.datafile,"r")

t=np.array(avgData["t"])
locs = t.argsort()
t=t[locs]

N=np.array(avgData["N_avg_mag"])[locs]
Nbar=np.array(avgData["Nbar_avg_mag"])[locs]
F=np.array(avgData["F_avg_mag"])[locs]
Fbar=np.array(avgData["Fbar_avg_mag"])[locs]
avgData.close()

# Calculate growth rate using a linear fit to the time interval
linear_interval = np.where((t >= args.time_start) & (t <= args.time_end))

t = np.squeeze(t[linear_interval])

# Take natural logs here since we want a base-e exponential growth rate
N = np.log(np.squeeze(N[linear_interval]))
Nbar = np.log(np.squeeze(Nbar[linear_interval]))
F = np.log(np.squeeze(F[linear_interval]))
Fbar = np.log(np.squeeze(Fbar[linear_interval]))

N_growth = np.zeros((3,3))
Nbar_growth = np.zeros((3,3))
F_growth = np.zeros((3,3,3))
Fbar_growth = np.zeros((3,3,3))

def save_growth(t, Q, growth_matrix):
    for i in range(3):
        for j in range(3):
            coef = np.polyfit(t, Q[:,i,j], 1)
            growth_matrix[i,j] = coef[0]
            growth_matrix[j,i] = coef[0]

save_growth(t, N, N_growth)
save_growth(t, Nbar, Nbar_growth)

for i in range(3):
    save_growth(t, F[:,i,:,:], F_growth[i,:,:])
    save_growth(t, Fbar[:,i,:,:], Fbar_growth[i,:,:])

if args.print:
    print("N growth rates:")
    print(N_growth)

    print("Nbar growth rates:")
    print(Nbar_growth)

    print("Fx growth rates:")
    print(F_growth[0,:,:])

    print("Fy growth rates:")
    print(F_growth[1,:,:])

    print("Fz growth rates:")
    print(F_growth[2,:,:])

    print("Fbarx growth rates:")
    print(Fbar_growth[0,:,:])

    print("Fbary growth rates:")
    print(Fbar_growth[1,:,:])

    print("Fbarz growth rates:")
    print(Fbar_growth[2,:,:])

# Write the output hdf5 file
output = h5py.File("{}.h5".format(args.output), "w")

# Save the arguments to this script we used
output["input_datafile"] = args.datafile
output["fit_start_time"] = args.time_start
output["fit_end_time"] = args.time_end

# Save the label for this data
output["label"] = args.label

# Save the growth rates
output["growth_N"] = N_growth
output["growth_F"] = F_growth
output["growth_Nbar"] = Nbar_growth
output["growth_Fbar"] = Fbar_growth

output.close()