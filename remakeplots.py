# Use the data output from makeplots to plot the domain-averaged number density and flux
# This way data can be re-used without waiting for the analysis every time it is run
# Run from the directory you want the pdf to be saved.
# Must have already run makeplots.py in the target simulation directory.

import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py

directories = sorted(glob.glob("plt*"))

base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]

def offdiagMag(f):
    R = 0
    I = 1
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)

def error_plot(t, q, qbar):
    plt.cla()
    print(np.shape(q))
    print(np.shape(qbar))
    tr    =    q[:,0,0] +    q[:,1,1] +    q[:,2,2]
    trbar = qbar[:,0,0] + qbar[:,1,1] + qbar[:,2,2]
    plt.plot(t, (tr[:,0] - tr[0,0])/tr[0,0],color="black",label=r"$\Sigma \mathrm{Tr}(f)$")
    plt.plot(t, (trbar[:,0] - trbar[0,0])/trbar[0,0],color="black",label=r"$\Sigma \mathrm{Tr}(\bar{f})$")
    plt.ylabel("error")
    plt.xlabel("t (s)")
    plt.legend()
    plt.savefig("trace.png")

def time_series_plots(q, qbar, name):

    # real/imag
    R=0
    I=1
    
    plt.cla()
    plt.plot(t, q[:,0,0,R],color="blue",label=r"$f_{00}$")
    plt.plot(t, q[:,1,1,R],color="red", label=r"$f_{11}$")
    plt.plot(t, q[:,2,2,R],color="gold",label=r"$f_{22}$")
    plt.plot(t, qbar[:,0,0,R],color="blue", linestyle="--",label=r"$\bar{f}_{00}$")
    plt.plot(t, qbar[:,1,1,R],color="red",  linestyle="--",label=r"$\bar{f}_{11}$")
    plt.plot(t, qbar[:,2,2,R],color="gold", linestyle="--",label=r"$\bar{f}_{22}$")
    plt.ylabel("f")
    plt.xlabel("t (s)")
    #plt.xscale("log")
    plt.ylim((.3,.4))
    plt.xlim((0,5e-7))
    plt.grid()
    plt.legend(ncol=2)
    plt.savefig(name+"diag.png")

    plt.clf()
    plt.plot(t, q[:,0,1,R],color="purple",label=r"$|f_{01}|$")
    plt.plot(t, q[:,0,2,R],color="green" ,label=r"$|f_{02}|$")
    plt.plot(t, q[:,1,2,R],color="orange",label=r"$|f_{12}|$")
    plt.plot(t, q[:,0,1,I],color="purple",alpha=0.5)
    plt.plot(t, q[:,0,2,I],color="green" ,alpha=0.5)
    plt.plot(t, q[:,1,2,I],color="orange",alpha=0.5)
    plt.plot(t, offdiagMag(q),color="black", label=r"$|f_\mathrm{offdiag}|$")
    plt.plot(t, qbar[:,0,1,R],color="purple", linestyle="--",label=r"$|\bar{f}_{01}|$")
    plt.plot(t, qbar[:,0,2,R],color="green",  linestyle="--",label=r"$|\bar{f}_{02}|$")
    plt.plot(t, qbar[:,1,2,R],color="orange", linestyle="--",label=r"$|\bar{f}_{12}|$")
    plt.plot(t, qbar[:,0,1,I],color="purple", linestyle="--",alpha=0.5)
    plt.plot(t, qbar[:,0,2,I],color="green",  linestyle="--",alpha=0.5)
    plt.plot(t, qbar[:,1,2,I],color="orange", linestyle="--",alpha=0.5)
    plt.plot(t, offdiagMag(qbar),color="black", linestyle="--", label=r"$|f_\mathrm{offdiag}|$")
    plt.ylabel("f")
    plt.xlabel("t (s)")
    #plt.xscale("log")
    plt.grid()
    plt.legend(ncol=2)
    plt.savefig(name+"offdiag.png")



# read averaged data
avgData = h5py.File("avg_data.h5","r")

t=np.array(avgData["t"])
locs = t.argsort()
t=t[locs]

N=np.array(avgData["N"])[locs]
Nbar=np.array(avgData["Nbar"])[locs]
F=np.array(avgData["F"])[locs]
Fbar=np.array(avgData["Fbar"])[locs]
avgData.close()
time_series_plots(N,Nbar,"N")
time_series_plots(F[:,0],Fbar[:,0],"Fx")
time_series_plots(F[:,1],Fbar[:,1],"Fy")
time_series_plots(F[:,2],Fbar[:,2],"Fz")
error_plot(t,N,Nbar)
