"""
1D relaxation exchange experiment

from:
    Van Landeghem M, Haber A, D’espinose De Lacaillerie J-B, Blümich B
    Analysis of multisite 2D relaxation exchange NMR. 
    Concepts Magn Reson 2010; 36A:153–169.
"""

import numpy as np
from matplotlib import pyplot as plt
from epgpy import operators, functions, exchange
from epgpy.utilities import ilt1d

# properties
T2a = 2.5
T2b = 25

# echange rate
Nrate = 20
rates = np.geomspace(1e-3, 10, Nrate)

# CPMG sequence
TE = 0.1  # ms
# TM = # mixing time
Necho = 200

adc = operators.ADC
exc = operators.T(90, 90)
rfc = operators.T(180, 0)
khi = exchange.exchange_matrix(rates, axis=1, ncomp=2)
xt = operators.X(TE / 2, khi, T2=[[T2a, T2b]], axis=1, duration=True)

seq = [exc] + [xt, rfc, xt, adc] * Necho

# simulate
times = functions.get_adc_times(seq)
sim = functions.simulate(seq)

# add signals
sig = 0.5 * (sim[..., 0] + sim[..., 1]).real

# Inverse laplace transform
_, kernel = ilt1d.get_kernel(times, [1e-3, 1e3], 20)
tsig = [ilt1d.ilt1d(times, sig[:, i], kernel=kernel, ls=True) for i in range(Nrate)]


plt.figure("exchange-ilt1d", figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, Nrate))


plt.subplot(2, 1, 1)
for i in range(Nrate):
    t, a = tsig[i]
    h = plt.scatter([rates[i]] * len(t), 1 / t, marker="d", color=colors[i])
    # plt.vlines([rates[i]] * len(t), np.min(1/t), np.max(1/t), color='k')
plt.legend(labels=["Apparent relaxation"], loc="upper right")
plt.xscale("log")
plt.grid()
plt.xlabel("Exchange rates (1/ms)")
plt.ylabel("Apparent relaxation times (ms)")
plt.title(f"Inverse Laplace transform (ARM+LS)")


plt.subplot(2, 2, 3)
for i in range(Nrate):
    plt.plot(times, sig[:, i], color=colors[i])
plt.ylabel("signal (a.u)")
plt.xlabel("time (ms)")
# plt.legend(labels=[f'{r:0.1}' for r in rates], title='Echange rate (1/ms)')
plt.title("Signal")
plt.yscale("log")

plt.subplot(2, 2, 4)
for i in range(Nrate):
    t, a = tsig[i]
    h = plt.scatter([rates[i]] * len(t), a, marker="d", color=colors[i])
# plt.legend(labels=['Component amplitude'], loc='upper left')
plt.xscale("log")
plt.grid()
plt.xlabel("Exchange rates (1/ms)")
plt.ylabel("Component amplitude (a.u)")
plt.title(f"ILT - component amplitude")
# plt.subplot(2,1,2)


plt.suptitle(f"Simulation of a 2-site exchange with T2={T2a}/{T2b}ms, A1=A2=0.5")

plt.tight_layout()
plt.show()
