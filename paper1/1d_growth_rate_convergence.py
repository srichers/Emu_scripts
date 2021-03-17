# Calculate growth rates for a "reduced_data" file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator,LogLocator)
import h5py
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datafiles", type=str, nargs="+", help="Set of data files containing growth rates.")
args = parser.parse_args()

# Read growth rate data
growth_rates = []

class GrowthRates(object):
    def __init__(self, gN=None, gNbar=None, gF=None, gFbar=None, resolution=None):
        self.growth_N = gN
        self.growth_Nbar = gNbar
        self.growth_F = gF
        self.growth_Fbar = gFbar
        self.resolution = resolution

re_nx = re.compile("[0-9]D_nx(?P<resolution>[0-9]+)")

for f in args.datafiles:
    d = h5py.File(f, "r")

    gr = GrowthRates()
    gr.growth_N = np.array(d["growth_N"])
    gr.growth_Nbar = np.array(d["growth_Nbar"])
    gr.growth_F = np.array(d["growth_F"])
    gr.growth_Fbar = np.array(d["growth_Fbar"])
    label = str(np.array(d["label"]))
    nx_match = re_nx.match(label)
    assert(nx_match is not None)
    gr.resolution = float(nx_match.group("resolution"))
    growth_rates.append(gr)

    d.close()

# Sort growth rates by resolution so the last has the highest resolution & smallest timestep
growth_rates = sorted(growth_rates, key=lambda x: x.resolution)

# Separate out reference point and the rest of the points for convergence
reference = growth_rates[-1]
growth_rates = growth_rates[:-1]

# Calculate log-scaled errors
errors = []

for g in growth_rates:
    e = GrowthRates()
    e.growth_N = np.abs(g.growth_N - reference.growth_N)
    e.growth_Nbar = np.abs(g.growth_Nbar - reference.growth_Nbar)
    e.growth_F = np.abs(g.growth_F - reference.growth_F)
    e.growth_Fbar = np.abs(g.growth_Fbar - reference.growth_Fbar)
    e.resolution = g.resolution
    errors.append(e)

# Get average convergence order for N using the last three points at large N
# (This averages over 2 error slopes)
orders = []
ordersF = []
num_slopes_average = 2
for i in range(num_slopes_average):
    i = len(errors) - i - 2
    eratioN = np.log10(errors[i+1].growth_N / errors[i].growth_N)
    eratioF = np.log10(errors[i+1].growth_F / errors[i].growth_F)
    eratioNbar = np.log10(errors[i+1].growth_Nbar / errors[i].growth_Nbar)
    eratioFbar = np.log10(errors[i+1].growth_Fbar / errors[i].growth_Fbar)
    rratio = np.log10(errors[i].resolution / errors[i+1].resolution)
    orders.append(eratioN / rratio)
    ordersF.append(eratioF / rratio)
orders = np.array(orders)
ordersF = np.array(ordersF)

order_average = np.average(orders, axis=0)
orderF_average = np.average(ordersF, axis=0)

# Plot errors as a function of resolution

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

fig, axs = plt.subplots(2,1,figsize=(6,12))
plt.subplots_adjust(hspace=0,wspace=0)

axs[0].xaxis.set_minor_locator(AutoMinorLocator())
axs[0].xaxis.set_major_locator(MultipleLocator(0.5))
axs[1].xaxis.set_minor_locator(AutoMinorLocator())
axs[1].xaxis.set_major_locator(MultipleLocator(0.5))
axs[0].set_ylim([0, 3.2])
axs[1].set_ylim([7.9, 10.99])

axs[0].yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
axs[1].yaxis.set_tick_params(which='both', direction='in', right=True,top=True)
axs[0].xaxis.set_tick_params(which='both', direction='in', right=True,top=True)
axs[1].xaxis.set_tick_params(which='both', direction='in', right=True,top=True)
axs[0].minorticks_on()
axs[1].minorticks_on()

resolution = np.array([e.resolution for e in errors])
timescale = reference.resolution / resolution
log_timescale = np.log10(timescale)
imin_log_ts = np.argmin(log_timescale)
min_log_ts = np.min(log_timescale)
max_log_ts = np.max(log_timescale)

def plot_nij(axis, i,j,label,color):
    # nij
    err_nij = np.array([e.growth_N[i,j] for e in errors])
    log_err_nij = np.log10(err_nij)
    scale = log_err_nij[imin_log_ts]
    axis.plot(log_timescale, log_err_nij - scale, marker='o', color=color, linestyle="None", label=label)

def plot_fxij(axis, i,j,label,color):
    # fxij
    err_fxij = np.array([e.growth_F[0,i,j] for e in errors])
    log_err_fxij = np.log10(err_fxij)
    axis.plot(log_timescale, log_err_fxij, marker='o', color=color, linestyle="None", label=label)

def plot_order(axis,order,label,color):
    x = np.linspace(min_log_ts, max_log_ts)
    y = order * (x - min_log_ts)
    axis.plot(x, y, marker="None", linestyle="--", color=color, label=label)

ax = axs[1]
plot_fxij(ax, 0,1, r"$\mathbf{f}^{(x)}_{e\mu}$", "purple")
plot_fxij(ax, 0,2, r"$\mathbf{f}^{(x)}_{e\tau}$", "green")
plot_fxij(ax, 1,2, r"$\mathbf{f}^{(x)}_{\mu\tau}$", "orange")
plot_fxij(ax, 1,1, r"$\mathbf{f}^{(x)}_{\mu\mu}$", "brown")
plot_fxij(ax, 2,2, r"$\mathbf{f}^{(x)}_{\tau\tau}$", "salmon")
x = np.linspace(min_log_ts, max_log_ts)
y = 2 * (x - 0.5) + 8.75
ax.plot(x, y, marker="None", linestyle="--", color="magenta", label="2nd order")

ax = axs[0]
plot_nij(ax, 0,1, r"$n_{e\mu}$", "purple")
plot_nij(ax, 0,2, r"$n_{e\tau}$", "green")
plot_nij(ax, 1,2, r"$n_{\mu\tau}$", "orange")
plot_nij(ax, 1,1, r"$n_{\mu\mu}$", "brown")
plot_nij(ax, 2,2, r"$n_{\tau\tau}$", "salmon")

plot_order(ax, 1, "1st order", "red")
plot_order(ax, 2, "2nd order", "magenta")
plot_order(ax, 3, "3rd order", "cyan")
plot_order(ax, 4, "4th order", "blue")

axs[0].set_xlabel("")
axs[1].set_xlabel(r"$\mathrm{log_{10}}(\Delta z / \Delta z_{2048})$")

axs[0].set_ylabel(r"$\mathrm{log_{10}}(\delta \omega) - \mathrm{min(log_{10}}(\delta \omega))$")
axs[1].set_ylabel(r"$\mathrm{log_{10}}(\delta \omega)$")

axs[0].legend(bbox_to_anchor=(1.02, -0.02), loc="lower right", frameon=False, ncol=2, fontsize=18, handlelength=0.8, handletextpad=0.25, columnspacing=.7)
axs[1].legend(bbox_to_anchor=(1.02, -0.02), loc="lower right", frameon=True, ncol=2, fontsize=18, handlelength=0.8, handletextpad=0.25, columnspacing=.7)

fig.align_xlabels(axs)
fig.align_ylabels(axs)

plt.savefig("convergence_order.pdf", bbox_inches="tight")

# def plot_neu():
#     # neu
#     err_neu = np.array([e.growth_N[0,1] for e in errors])
#     log_err_neu = np.log10(err_neu)
#     scale = np.min(log_err_neu)
#     ax.plot(log_timescale, log_err_neu/scale, marker='o', linestyle="None", label="Neu")

#     order_neu = order_average[0,1]
#     iMinErr = np.argmin(log_err_neu)
#     intercept = log_err_neu[iMinErr] - order_neu * log_timescale[iMinErr]
#     log_order_err = intercept + order_neu * log_timescale
#     num_points_slope = num_slopes_average + 1
#     ax.plot(log_timescale[-num_points_slope:], log_order_err[-num_points_slope:]/scale, marker="None", linestyle="--", label="O = {}".format(order_neu))
#     print("order neu: ", order_neu)

# def plot_nuu():
#     # nuu
#     err_nuu = np.array([e.growth_N[1,1] for e in errors])
#     log_err_nuu = np.log10(err_nuu)
#     scale = np.min(log_err_nuu)
#     ax.plot(log_timescale, log_err_nuu/scale, marker='o', linestyle="None", label="Nuu")

#     order_nuu = order_average[1,1]
#     iMinErr = np.argmin(log_err_nuu)
#     intercept = log_err_nuu[iMinErr] - order_nuu * log_timescale[iMinErr]
#     log_order_err = intercept + order_nuu * log_timescale
#     num_points_slope = num_slopes_average + 1
#     ax.plot(log_timescale[-num_points_slope:], log_order_err[-num_points_slope:]/scale, marker="None", linestyle="--", label="O = {}".format(order_nuu))
#     print("order nuu: ", order_nuu)

# def plot_ntt():
#     # ntt
#     err_ntt = np.array([e.growth_N[2,2] for e in errors])
#     log_err_ntt = np.log10(err_ntt)
#     scale = np.min(log_err_ntt)
#     ax.plot(log_timescale, log_err_ntt/scale, marker='o', linestyle="None", label="Ntt")

#     order_ntt = order_average[2,2]
#     iMinErr = np.argmin(log_err_ntt)
#     intercept = log_err_ntt[iMinErr] - order_ntt * log_timescale[iMinErr]
#     log_order_err = intercept + order_ntt * log_timescale
#     num_points_slope = num_slopes_average + 1
#     ax.plot(log_timescale[-num_points_slope:], log_order_err[-num_points_slope:]/scale, marker="None", linestyle="--", label="O = {}".format(order_ntt))
#     print("order ntt: ", order_ntt)

# def plot_nut():
#     # nut
#     err_nut = np.array([e.growth_N[1,2] for e in errors])
#     log_err_nut = np.log10(err_nut)
#     scale = np.min(log_err_nut)
#     ax.plot(log_timescale, log_err_nut/scale, marker='o', linestyle="None", label="Nut")

#     order_nut = order_average[1,2]
#     iMinErr = np.argmin(log_err_nut)
#     intercept = log_err_nut[iMinErr] - order_nut * log_timescale[iMinErr]
#     log_order_err = intercept + order_nut * log_timescale
#     num_points_slope = num_slopes_average + 1
#     ax.plot(log_timescale[-num_points_slope:], log_order_err[-num_points_slope:]/scale, marker="None", linestyle="--", label="O = {}".format(order_nut))
#     print("order nut: ", order_nut)
