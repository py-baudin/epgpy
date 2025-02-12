import numpy as np
from matplotlib import pyplot as plt
import epgpy as epg
from epgpy import stats

"""
# Toy example 1: optimization of MSE sequence for a single tissue

Example taken from:

> Lee PK, Watkins LE, Anderson TI, Buonincontri G, Hargreaves BA:
  Flexible and Efficient Optimization of Quantitative Sequences using Automatic
  Differentiation of Bloch Simulations.
  Magn Reson Med 2019; 82:1438–1451.

"""

# constants
T1 = 1400
T2 = 10
necho = 6
# multiple tau values
tau = [1, 5, 10, 20]

# define virtual operators
rlx = epg.E(tau, T1, T2)
exc = epg.T(90, 90)
inv = epg.T(180, 0)
grd = epg.S(1)
adc = epg.ADC

# define sequence as function of tau
seq = [exc] + [grd, rlx, inv, grd, rlx, adc] * necho

# simulate sequence for necho=6 and various tau values
signal = epg.simulate(seq)

# plot seq for various echo-spacing
plt.figure("seq-mse")
indices = np.arange(len(signal)) + 1
lines = plt.plot(indices, np.abs(signal))
plt.title(f"MSE: signal vs echo spacing (T2=10ms)")
plt.legend(lines, map(str, tau), title="tau values (ms)")
plt.grid()
plt.ylabel("signal")
plt.xlabel("echo index")


"""
Here, we compute and plot the CRLB of T2, ie a lower bound on the variance of an estimate of T2,
with respect to the sequence parameters (tau), and a given noise level.

For a given number of echos, and noise level (sigma1=1), we find a (unique) minimum.

Example: for necho=1, the optimum tau is 1/2, ie. the inter echo spacing (ESP) is equal to T2.

"""

plt.figure("seq-mse-crlb")
tau = np.linspace(0.5, 10, 1000)
rlx = epg.E(tau, T1, T2, order1="T2")
colors = {}
for necho in range(1, 7):
    seq = [exc] + [grd, rlx, inv, grd, rlx, adc] * necho
    jac = epg.simulate(seq, probe=epg.Jacobian("T2"))
    # log10(CRB) of the MSE sequence
    cost = stats.crlb(np.moveaxis(jac, -2, 0), log=True, W=[10])
    # index of optimal tau value
    argmin = np.argmin(cost)
    h = plt.plot(2 * tau / 10, cost, label=f"Nechos={necho}")
    plt.scatter(2 * tau[argmin] / 10, cost[argmin])
    colors[necho] = h[0].get_color()

ylim = plt.ylim()
plt.title("MSE: CRLB for fixed T2 (10ms) with known $S_0$")
plt.xlabel("ESP/T2")
plt.ylabel("log10(CRLB)")
plt.grid()
plt.legend()

plt.figure("seq-mse-crlb-s0")
tau = np.linspace(0.5, 10, 1000)
rlx = epg.E(tau, T1, T2, order1="T2")

for necho in range(2, 7):
    seq = [exc] + [grd, rlx, inv, grd, rlx, adc] * necho
    jac = epg.simulate(seq, probe=epg.Jacobian(["magnitude", "T2"]))
    # log10(CRB) of the MSE sequence
    cost = stats.crlb(np.moveaxis(jac, -2, 0), log=True, W=[1, 10])
    # index of optimal tau value
    argmin = np.argmin(cost)
    plt.plot(2 * tau / 10, cost, color=colors[necho], label=f"Nechos={necho}")
    plt.scatter(2 * tau[argmin] / 10, cost[argmin], color=colors[necho])

plt.ylim(ylim)
plt.title("MSE: CRLB for fixed T2 (10ms) with unknown $S_0$")
plt.xlabel("ESP/T2")
plt.ylabel("log10(CRLB)")
plt.grid()
plt.legend()

plt.show()
