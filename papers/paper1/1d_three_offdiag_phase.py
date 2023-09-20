# Generate the 3x3 plots of 1d flavor off-diagonal complex phases.
# run from the directory you want the pdf to be saved

import numpy as np
import matplotlib.pyplot as plt
import yt
import glob
import multiprocessing as mp
import h5py
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)

basedir="/global/project/projectdirs/m3018/Emu/PAPER/1D/1D_fiducial"
directories = [basedir+"/plt00000",basedir+"/plt00300",basedir+"/plt04840"]

base=["N","Fx","Fy","Fz"]
diag_flavor=["00","11","22"]
offdiag_flavor=["01","02","12"]
re=["Re","Im"]

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



# assumes phi is in radians/pi
def phase_plot(z, phi, color, linestyle, ax,label=None):
    dphi = phi[1:]-phi[:-1]
    b = np.where(np.abs(dphi)>1)[0]+1 # breakpoints
    b = np.append(b, len(phi))
    b = np.append([0],b)

    for ib in range(len(b)-1):
        if(ib==0):
            this_label = label
        else:
            this_label = None
        iL = b[ib]
        iR = b[ib+1]
        ax.plot(z[iL:iR], phi[iL:iR], color=color, linestyle=linestyle,label=this_label)
            

def snapshot_plot(ad, axes, t, components=None):
    # components can either be None, in which case all off diagonal
    # components are plotted, or a list containing any subset of the following:
    # - "emu": plots e-mu neutrino components
    # - "etau": plots e-tau neutrino components
    # - "mutau": plots mu-tau neutrino components
    # - "emubar": plots e-mu anti-neutrino components
    # - "etaubar": plots e-tau anti-neutrino components
    # - "mutaubar": plots mu-tau anti-neutrino components

    z = ad['index',"z"]    

    trace    = ad['boxlib',"N00_Re"   ] + ad['boxlib',"N11_Re"   ] + ad['boxlib',"N22_Re"   ]
    tracebar = ad['boxlib',"N00_Rebar"] + ad['boxlib',"N11_Rebar"] + ad['boxlib',"N22_Rebar"]

    N01 = ad['boxlib',"N01_Re"] + 1j*ad['boxlib',"N01_Im"]
    N02 = ad['boxlib',"N02_Re"] + 1j*ad['boxlib',"N02_Im"]
    N12 = ad['boxlib',"N12_Re"] + 1j*ad['boxlib',"N12_Im"]
    N01bar = ad['boxlib',"N01_Rebar"] + 1j*ad['boxlib',"N01_Imbar"]
    N02bar = ad['boxlib',"N02_Rebar"] + 1j*ad['boxlib',"N02_Imbar"]
    N12bar = ad['boxlib',"N12_Rebar"] + 1j*ad['boxlib',"N12_Imbar"]

    Fx01 = ad['boxlib',"Fx01_Re"] + 1j*ad['boxlib',"Fx01_Im"]
    Fx02 = ad['boxlib',"Fx02_Re"] + 1j*ad['boxlib',"Fx02_Im"]
    Fx12 = ad['boxlib',"Fx12_Re"] + 1j*ad['boxlib',"Fx12_Im"]
    Fx01bar = ad['boxlib',"Fx01_Rebar"] + 1j*ad['boxlib',"Fx01_Imbar"]
    Fx02bar = ad['boxlib',"Fx02_Rebar"] + 1j*ad['boxlib',"Fx02_Imbar"]
    Fx12bar = ad['boxlib',"Fx12_Rebar"] + 1j*ad['boxlib',"Fx12_Imbar"]

    Fy01 = ad['boxlib',"Fy01_Re"] + 1j*ad['boxlib',"Fy01_Im"]
    Fy02 = ad['boxlib',"Fy02_Re"] + 1j*ad['boxlib',"Fy02_Im"]
    Fy12 = ad['boxlib',"Fy12_Re"] + 1j*ad['boxlib',"Fy12_Im"]
    Fy01bar = ad['boxlib',"Fy01_Rebar"] + 1j*ad['boxlib',"Fy01_Imbar"]
    Fy02bar = ad['boxlib',"Fy02_Rebar"] + 1j*ad['boxlib',"Fy02_Imbar"]
    Fy12bar = ad['boxlib',"Fy12_Rebar"] + 1j*ad['boxlib',"Fy12_Imbar"]

    Fz01 = ad['boxlib',"Fz01_Re"] + 1j*ad['boxlib',"Fz01_Im"]
    Fz02 = ad['boxlib',"Fz02_Re"] + 1j*ad['boxlib',"Fz02_Im"]
    Fz12 = ad['boxlib',"Fz12_Re"] + 1j*ad['boxlib',"Fz12_Im"]
    Fz01bar = ad['boxlib',"Fz01_Rebar"] + 1j*ad['boxlib',"Fz01_Imbar"]
    Fz02bar = ad['boxlib',"Fz02_Rebar"] + 1j*ad['boxlib',"Fz02_Imbar"]
    Fz12bar = ad['boxlib',"Fz12_Rebar"] + 1j*ad['boxlib',"Fz12_Imbar"]

    Fperp00    = np.sqrt(ad['boxlib',"Fx00_Re"   ]**2+ad['boxlib',"Fy00_Re"   ]**2)
    Fperp11    = np.sqrt(ad['boxlib',"Fx11_Re"   ]**2+ad['boxlib',"Fy11_Re"   ]**2)
    Fperp22    = np.sqrt(ad['boxlib',"Fx22_Re"   ]**2+ad['boxlib',"Fy22_Re"   ]**2)
    Fperp00bar = np.sqrt(ad['boxlib',"Fx00_Rebar"]**2+ad['boxlib',"Fy00_Rebar"]**2)
    Fperp11bar = np.sqrt(ad['boxlib',"Fx11_Rebar"]**2+ad['boxlib',"Fy11_Rebar"]**2)
    Fperp22bar = np.sqrt(ad['boxlib',"Fx22_Rebar"]**2+ad['boxlib',"Fy22_Rebar"]**2)

    Fperp01    = np.sqrt(Fx01**2 + Fy01**2)
    Fperp02    = np.sqrt(Fx02**2 + Fy02**2)
    Fperp12    = np.sqrt(Fx12**2 + Fy12**2)
    Fperp01bar = np.sqrt(Fx01bar**2 + Fy01bar**2)
    Fperp02bar = np.sqrt(Fx02bar**2 + Fy02bar**2)
    Fperp12bar = np.sqrt(Fx12bar**2 + Fy12bar**2)
    
    ax = axes[0]
    if components is None or "emu" in components:
        phase_plot(z, np.angle(N01   )/np.pi, color="purple", linestyle="-" , ax=ax, label=r"${e\mu}$"         )
    if components is None or "etau" in components:
        phase_plot(z, np.angle(N02   )/np.pi, color="green" , linestyle="-" , ax=ax, label=r"${e\tau}$"        )
    if components is None or "mutau" in components:
        phase_plot(z, np.angle(N12   )/np.pi, color="orange", linestyle="-" , ax=ax, label=r"${\mu\tau}$"      )
    if components is None or "emubar" in components:
        phase_plot(z, np.angle(N01bar)/np.pi, color="purple", linestyle="--", ax=ax, label=r"$\bar{e\mu}$"   )
    if components is None or "etaubar" in components:
        phase_plot(z, np.angle(N02bar)/np.pi, color="green" , linestyle="--", ax=ax, label=r"$\bar{e\tau}$"  )
    if components is None or "mutaubar" in components:
        phase_plot(z, np.angle(N12bar)/np.pi, color="orange", linestyle="--", ax=ax, label=r"$\bar{\mu\tau}$")

    #ax = axes[1]
    #phase_plot(z, np.angle(Fz01   )/np.pi, color="purple", linestyle="-" , ax=ax, label=r"$f_{e\mu}$"         )
    #phase_plot(z, np.angle(Fz02   )/np.pi, color="green" , linestyle="-" , ax=ax, label=r"$f_{e\tau}$"        )
    #phase_plot(z, np.angle(Fz12   )/np.pi, color="orange", linestyle="-" , ax=ax, label=r"$f_{\mu\tau}$"      )
    #phase_plot(z, np.angle(Fz01bar)/np.pi, color="purple", linestyle="--", ax=ax, label=r"$\bar{f}_{e\mu}$"   )
    #phase_plot(z, np.angle(Fz02bar)/np.pi, color="green" , linestyle="--", ax=ax, label=r"$\bar{f}_{e\tau}$"  )
    #phase_plot(z, np.angle(Fz12bar)/np.pi, color="orange", linestyle="--", ax=ax, label=r"$\bar{f}_{\mu\tau}$")

    ax = axes[1]
    if components is None or "emu" in components:
        phase_plot(z, np.angle(Fx01   )/np.pi, color="purple", linestyle="-" , ax=ax, label=r"${e\mu}$"         )
    if components is None or "etau" in components:
        phase_plot(z, np.angle(Fx02   )/np.pi, color="green" , linestyle="-" , ax=ax, label=r"${e\tau}$"        )
    if components is None or "mutau" in components:
        phase_plot(z, np.angle(Fx12   )/np.pi, color="orange", linestyle="-" , ax=ax, label=r"${\mu\tau}$"      )
    if components is None or "emubar" in components:
        phase_plot(z, np.angle(Fx01bar)/np.pi, color="purple", linestyle="--", ax=ax, label=r"$\bar{e\mu}$"   )
    if components is None or "etaubar" in components:
        phase_plot(z, np.angle(Fx02bar)/np.pi, color="green" , linestyle="--", ax=ax, label=r"$\bar{e\tau}$"  )
    if components is None or "mutaubar" in components:
        phase_plot(z, np.angle(Fx12bar)/np.pi, color="orange", linestyle="--", ax=ax, label=r"$\bar{\mu\tau}$")

    
if __name__ == "__main__":
    t=[]

    fig, axes = plt.subplots(2,3, figsize=(16,6))
    plt.subplots_adjust(hspace=0,wspace=0.05)

    for i,d in zip(range(3),directories):
        print(d)
        ds = yt.load(d)
        t.append(ds.current_time)
        ad = ds.all_data()

        snapshot_plot(ad, axes[:,i], t[-1])

    xmax = 16
    for ax in axes.flatten():
        ax.tick_params(axis='both', which='both', direction='in', right=True,top=True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(0,xmax*.999)
    for ax in axes[:,2]:
        ax.set_xlim(0,xmax)
        
    alpha = 0.9
    for i,ax in zip(range(3),axes[0,:]):
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_xticklabels([])
        ax.set_ylim(-.05,1.05)
        ax.text(8,.75, "t=%0.2f ns"%(t[i]*1e9), ha="center", va="center",bbox=dict(facecolor='white', alpha=alpha, edgecolor='.75'))


    ylabel = r"$\phi_{n_{ab}}$"
    axes[0,0].set_ylabel(ylabel)    
    axes[0,0].set_yticklabels(["",0,r"$\pi$"])
    for ax in axes[0,:]:
        ax.set_ylim(-1,1)
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_xticklabels([])
        ax.set_xlabel(r"$x\,(\mathrm{cm})$")

    #ylabel = r"$\phi_{F^{(z)}_{ij}}$"
    #axes[1,0].set_ylabel(ylabel)    
    #for ax in axes[1,:]:
    #    ax.set_ylim(-1,1)
    #    ax.yaxis.set_minor_locator(MultipleLocator(.25))
    #    ax.set_xticklabels([])
    #    ax.set_xlabel(r"$x\,(\mathrm{cm})$")

    ylabel = r"$\phi_{\mathbf{f}^{(x)}_{ab}}$"
    axes[1,0].set_ylabel(ylabel)    
    axes[1,0].set_yticklabels([r"$-\pi$",0,r"$\pi$"])
    for ax in axes[1,:]:
        ax.set_ylim(-1,1)
        ax.yaxis.set_minor_locator(MultipleLocator(.25))
        ax.set_xlabel(r"$z\,(\mathrm{cm})$")

    for ax in axes[:,1:].flatten():
        ax.set_yticklabels([])


        
    axes[1,0].legend(frameon=True,loc=10,ncol=2,framealpha=alpha)
    #for ax in axes
    #ax.set_xlabel(r"$z$ (cm)")
    #ax.set_ylabel(r"$N$ (cm$^{-3 }$)")
    #ax.set_ylabel(r"$N$ (cm$^{-3 }$)")

    fig.align_xlabels(axes)
    fig.align_ylabels(axes)
    plt.savefig("1d_three_offdiag_phase.pdf",bbox_inches="tight")
