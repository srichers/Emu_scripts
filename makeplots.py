# used to make plots but now just generates a hdf5 file with domain-averaged data.
# Run in the directory of the simulation the data should be generated for.
# Still has functionality for per-snapshot plots, but the line is commented out.

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

def get_matrix(base,suffix):
    f00  = ad['boxlib',base+"00_Re"+suffix]
    f01  = ad['boxlib',base+"01_Re"+suffix]
    f01I = ad['boxlib',base+"01_Im"+suffix]
    f02  = ad['boxlib',base+"02_Re"+suffix]
    f02I = ad['boxlib',base+"02_Im"+suffix]
    f11  = ad['boxlib',base+"11_Re"+suffix]
    f12  = ad['boxlib',base+"12_Re"+suffix]
    f12I = ad['boxlib',base+"12_Im"+suffix]
    f22  = ad['boxlib',base+"22_Re"+suffix]
    zero = np.zeros(np.shape(f00))
    fR = [[f00 , f01 , f02 ], [f01 ,f11 ,f12 ], [f02 ,f12 ,f22 ]]
    fI = [[zero, f01I, f02I], [f01I,zero,f12I], [f02I,f12I,zero]]
    return fR, fI

def sumtrace_N(N):
    sumtrace = np.sum(N[0][0]+N[1][1]+N[2][2])
    return sumtrace

def averaged_N(N, NI, sumtrace):
    R=0
    I=1

    # do the averaging
    # f1, f2, R/I
    Nout = np.zeros((3,3,2))
    for i in range(3):
        for j in range(3):
            Nout[i][j][R] = float(np.sum( N[i][j]) / sumtrace)
            Nout[i][j][I] = float(np.sum(NI[i][j]) / sumtrace)

    return np.array(Nout)

def averaged_F(F, FI):
    R=0
    I=1
    
    # get spatial vector length
    Fmag = np.sqrt(F[0]**2 + F[1]**2 + F[2]**2)
    Fmagtrace = Fmag[0][0]+Fmag[1][1]+Fmag[2][2]
    
    # do the averaging
    # direction, f1, f2, R/I
    Fout = np.zeros((3,3,3,2))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                Fout[i][j][k][R] = float(np.sum( F[i][j][k])/np.sum(Fmagtrace))
                Fout[i][j][k][I] = float(np.sum(FI[i][j][k])/np.sum(Fmagtrace))

    return Fout

def offdiagMag(f):
    R = 0
    I = 1
    return np.sqrt(f[:,0,1,R]**2 + f[:,0,1,I]**2 +
                   f[:,0,2,R]**2 + f[:,0,2,I]**2 +
                   f[:,1,2,R]**2 + f[:,1,2,I]**2)

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
    plt.legend(ncol=2)
    plt.savefig(name+"offdiag.png")


def error_plot(q, qbar, name):
    plt.cla()
    plt.plot(t, (q   -   q[0])/   q[0],color="black",label=r"$\Sigma \mathrm{Tr}(f)$")
    plt.plot(t, (qbar-qbar[0])/qbar[0],color="black", linestyle="--",label=r"$\Sigma \mathrm{Tr}(\bar{f})$")
    plt.ylabel("error")
    plt.xlabel("t (s)")
    plt.legend()
    plt.savefig("trace.png")


def snapshot_plot(ad, t):
    plt.clf()
    z = ad['index',"z"]
    
    plt.plot(z, np.sqrt(ad['boxlib',"N01_Re"   ]**2+ad['boxlib',"N01_Im"   ]**2),
             color="purple", label=r"$|f_{01}|$")
    plt.plot(z, np.sqrt(ad['boxlib',"N02_Re"   ]**2+ad['boxlib',"N02_Im"   ]**2),
             color="green", label=r"$|f_{02}|$")
    plt.plot(z, np.sqrt(ad['boxlib',"N12_Re"   ]**2+ad['boxlib',"N12_Im"   ]**2),
             color="orange", label=r"$|f_{12}|$")
    plt.plot(z, np.sqrt(ad['boxlib',"N01_Rebar"]**2+ad['boxlib',"N01_Imbar"]**2),
             color="purple", linestyle="--", label=r"$|\bar{f}_{01}|$")
    plt.plot(z, np.sqrt(ad['boxlib',"N02_Rebar"]**2+ad['boxlib',"N02_Imbar"]**2),
             color="green", linestyle="--", label=r"$|\bar{f}_{02}|$")
    plt.plot(z, np.sqrt(ad['boxlib',"N12_Rebar"]**2+ad['boxlib',"N12_Imbar"]**2),
             color="orange", linestyle="--", label=r"$|\bar{f}_{12}|$")
    plt.xlabel(r"$z$ (cm)")
    plt.ylabel(r"$N$ (cm$^{-3 }$)")
    plt.legend(ncol=2)
    plt.title(r"$t=$"+"%.3g"%float(t)+" (s)")
    plt.savefig("Noffdiag_"+d+".png")

    plt.clf()
    z = ad['index',"z"]
    plt.plot(z, ad['boxlib',"N00_Re"   ], color="blue", label=r"$|f_{00}|$")
    plt.plot(z, ad['boxlib',"N11_Re"   ], color="red" , label=r"$|f_{11}|$")
    plt.plot(z, ad['boxlib',"N22_Re"   ], color="gold", label=r"$|f_{22}|$")
    plt.plot(z, ad['boxlib',"N00_Rebar"], color="blue",
             linestyle="--", label=r"$|\bar{f}_{00}|$")
    plt.plot(z, ad['boxlib',"N11_Rebar"], color="red",
             linestyle="--", label=r"$|\bar{f}_{11}|$")
    plt.plot(z, ad['boxlib',"N22_Rebar"], color="gold",
             linestyle="--", label=r"$|\bar{f}_{22}|$")
    plt.xlabel(r"$z$ (cm)")
    plt.ylabel(r"$N$ (cm$^{-3 }$)")
    plt.legend(ncol=2)
    plt.title(r"$t=$"+"%.3g"%float(t)+" (s)")
    plt.savefig("Ndiag_"+d+".png")

    
trace = []
N=[]
F=[]
tracebar = []
Nbar=[]
Fbar=[]
t=[]

for d in directories:
    print(d)
    ds = yt.load(d)
    t.append(ds.current_time)
    ad = ds.all_data()

    #snapshot_plot(ad, t[-1])

    thisN, thisNI = get_matrix("N",""   )
    sumtrace = sumtrace_N(thisN)
    trace.append(sumtrace)
    N.append(   averaged_N(thisN,thisNI,sumtrace))

    thisN, thisNI = get_matrix("N","bar")
    sumtrace = sumtrace_N(thisN)
    tracebar.append(sumtrace)
    Nbar.append(averaged_N(thisN,thisNI,sumtrace))

    thisFx, thisFxI = get_matrix("Fx","")
    thisFy, thisFyI = get_matrix("Fy","")
    thisFz, thisFzI = get_matrix("Fz","")
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    F.append(averaged_F(Ftmp, FtmpI))
    
    thisFx, thisFxI = get_matrix("Fx","bar") 
    thisFy, thisFyI = get_matrix("Fy","bar") 
    thisFz, thisFzI = get_matrix("Fz","bar") 
    Ftmp  = np.array([thisFx , thisFy , thisFz ])
    FtmpI = np.array([thisFxI, thisFyI, thisFzI])
    Fbar.append(averaged_F(Ftmp, FtmpI))
    

trace = np.array(trace)
N = np.array(N)
F = np.array(F)
tracebar = np.array(tracebar)
Nbar = np.array(Nbar)
Fbar = np.array(Fbar)

# write averaged data
avgData = h5py.File("avg_data.h5","w")
avgData["N"] = N
avgData["Nbar"] = Nbar
avgData["F"] = F
avgData["Fbar"] = Fbar
avgData["t"] = t
avgData.close()

#time_series_plots(N,Nbar,"N")
#time_series_plots(F[:,0],Fbar[:,0],"Fx")
#time_series_plots(F[:,1],Fbar[:,1],"Fy")
#time_series_plots(F[:,2],Fbar[:,2],"Fz")
#error_plot(trace, tracebar, "trace")
