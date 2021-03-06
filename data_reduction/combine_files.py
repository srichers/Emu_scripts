import glob
import h5py
import numpy as np
import sys

if(len(sys.argv) != 2):
    print()
    print("Usage: [combine_files.py filename], where filename is contained in each of the run* subdirectories")
    print()
    exit()

base_filename = sys.argv[1]
print(base_filename)

file_list = sorted(glob.glob("*/"+base_filename))

# get the number of datasets in the file
f = h5py.File(file_list[0],"r")
keylist = [key for key in f.keys()]
ndatasets = len(keylist)
f.close()

# collect the data in appended arrays
print()
datasets = [[] for i in range(ndatasets)]
for filename in file_list:
    print("getting data from",filename)
    f = h5py.File(filename,"r")
    for i, key in enumerate(keylist):
        datasets[i].append(np.array(f[key]))
    f.close()

# concatenate the arrays together
# output to file
print()
print("Outputting datasets to "+base_filename)
f = h5py.File(base_filename,"w")
for i, key in enumerate(keylist):
    if key=="k":
        datasets[i] = datasets[i][0]
    else:
        datasets[i] = np.concatenate(datasets[i], axis=0)
    f[key] = datasets[i]
    print(key, "\t",np.shape(datasets[i]) )
f.close()
