# Run analysis for all of the data directories in the list. Run from anywhere


import os
from shutil import copyfile
import subprocess
import time
from multiprocessing import Pool
import glob

scriptname="/global/project/projectdirs/m3018/Emu/PAPER/scripts/reduce_data.py"
nprocs=2

data_base = "/global/project/projectdirs/m3018/Emu/PAPER"
data_directories =[[data_base+"/1D/1D_fiducial"]]
#data_directories.append(glob.glob(data_base+"/1D/converge_direction/1D*"))
#data_directories.append(glob.glob(data_base+"/1D/converge_domain/1D*"))
#data_directories.append(glob.glob(data_base+"/1D/converge_nx/1D*"))
#data_directories.append(glob.glob(data_base+"/1D/nbar_dens/*"))
#data_directories.append(glob.glob(data_base+"/1D/fbar_direction/*"))
#data_directories.append(glob.glob(data_base+"/1D/fbar_fluxfac/*"))
data_directories.append(glob.glob(data_base+"/1D/rando_test/*"))
#data_directories.append(glob.glob(data_base+"/1D/matter/*"))
data_directories = [x for y in data_directories for x in y]
print(data_directories)
print()

def runAnalysis(dirname):
    print(dirname)
    os.chdir(dirname)
    logfile = open("analysis_output.txt", "w")
    subprocess.run(["python3",scriptname], stdout=logfile, stderr=logfile)
    logfile.close()


p = Pool(nprocs)
p.map(runAnalysis, data_directories)
