""" T2 star simulation

Compare T2' simulations using
- n-isochromats 
- phase state approach
- exponential function

"""

import numpy as np
import epgpy as epg
from matplotlib import pyplot as plt

# T2 prime
t2p = 5 # ms

# sample isochromats off-resonance frequencies according to the Cauchy law
niso = 10001
offres = np.tan(0.999 * np.pi * np.random.uniform(-0.5, 0.5, niso)) / 2 / np.pi
# offres = np.tan(0.999 * np.pi * np.linspace(-0.5, 0.5, niso)) / 2 / np.pi

N = 20
delta = 0.5
adc = epg.ADC
rf = epg.T(30, 90)
wait = epg.P(delta, 1/t2p * offres)

seq_iso = [rf] + [[wait, adc]] * N
sim_iso = epg.simulate(seq_iso).sum(-1) / niso

wait = epg.C(delta/t2p) # phase accumulation 
seq_epg = [rf] + [[wait, adc]] * N
sim_epg, tau = epg.simulate(seq_epg, kgrid=0.1, probe=('F', 't'))
sim_epg = (sim_epg * np.exp(-np.abs(tau))).sum((-2, -1))


plt.figure('t2-star')
plt.plot(np.abs(sim_iso), '--', label=f'isochromats (n={niso})')
plt.plot(np.abs(sim_epg), ':', label='Phase states (EPG)')
plt.plot(0.5 * np.exp(-delta / t2p * np.arange(1, N + 1)), 'o', label='exponential function')
plt.legend()
plt.show()


