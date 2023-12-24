import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#==============#
# plot options #
#==============#
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

data = np.genfromtxt("reduced0D.dat").transpose()

# note, numpy index starts at 0, so subtract 1 from header value
t = data[1]*1e9
Nee    = data[4] / 1e32
Neebar = data[5] / 1e32
Nmm    = data[12]/ 1e32
Nmmbar = data[13]/ 1e32
Ntt    = data[20]/ 1e32
Nttbar = data[21]/ 1e32

#==========#
# subplots #
#==========#
fig,ax=plt.subplots(1,1, figsize=(8,6))
plt.subplots_adjust(wspace=0, hspace=0)
fig.align_labels()

ax.plot(t, Nee, color="blue", linestyle="-", label=r"$n_{\nu_e}$")
ax.plot(t, Neebar, color="blue", linestyle="--", label=r"$n_{\bar{\nu}_e}$")
ax.plot(t, Nmm, color="green", linestyle="-", label=r"$n_{\nu_\mu}=n_{\bar{\nu}_\mu}$")
ax.plot(t, Ntt, color="red", linestyle="-", label=r"$n_{\nu_\tau}=n_{\bar{\nu}_\tau}$")

legend = ax.legend(frameon=False, labelspacing=0.1,fontsize=22, loc="upper right")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("$n$ ($10^{32}$ g cm$^{-3}$)")
ax.set_xlim(0,t[-1])

#=================#
# Tick Formatting #
#=================#
ax.tick_params(axis='both',which="both", direction="in",top=True,right=True)
ax.minorticks_on()

plt.savefig("flavortrans.pdf", bbox_inches="tight")
