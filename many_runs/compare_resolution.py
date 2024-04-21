import numpy as np
import h5py
import matplotlib.pyplot as plt

basedirlist = ["RUN_lowres_sqrt2",
               "RUN_standard",
               "RUN_standard_3F"]

data_filename = "/i136/j100/k080/allData.h5"

for basedir in basedirlist:
    f = h5py.File(basedir+data_filename,"r")
    N00 = np.average(np.array(f["N00_Re"]), axis=1)
    t = np.array(f["t(s)"])
    plt.plot(t, N00/N00[0], label=basedir)
    f.close()

plt.legend()
plt.savefig("compare_resolution.pdf",bbox_inches='tight')
