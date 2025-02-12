"""
Diffusion sensitivity of a RARE sequence with low constant flip angles

from
> Weigel M, Schwenk S, Kiselev VG, Scheffler K, Hennig J:
  Extended phase graphs with anisotropic diffusion. Journal of Magnetic Resonance 2010; 205:276–285.

"""

import numpy as np
from matplotlib import pyplot as plt
from epgpy import operators, functions, utils


ESP = 8  # ms (echo-spacing)
ETL = 11  # echo-train length
taurf = 2.56  # ms excitation/refocussing

# read dephase and spoiler
GR1 = 15  # mT/m
tau1 = 1.44  # ms
k1 = utils.get_wavenumber(tau1, GR1)

# read encoding
GR2 = 7.2  # mT/m
tau2 = 4.0  # ms
k2 = utils.get_wavenumber(tau2, GR2)

# spoilers
GS = 9.9  # mT/m
tauS = 0.72  # ms
kS = utils.get_wavenumber(tauS, GS)

# tissue properties
T1 = 1e3  # ms
T2 = 1e2  # ms
D = 1e-3  # mm^2/s

# sequence
exc = operators.T(90, 90)

# s1 = operators.S(k1)
s1 = operators.S(k2 / 2 + kS)
# d1 = operators.D(tau1, D, k=k1)
d1 = operators.D(tau1, D, k=k2 / 2 + kS)
e1 = operators.E(tau1, T1, T2)

s2 = operators.S(k2 / 2)
d2 = operators.D(tau2 / 2, D, k=k2 / 2)
e2 = operators.E(tau2 / 2, T1, T2)

sS = operators.S(kS)
dS = operators.D(tauS, D, k=kS)
eS = operators.E(tauS, T1, T2)

# rf pulses
angles = np.linspace(0, 180, 181)
trf = operators.T(angles, 0)
erf = operators.E(taurf / 2, T1, T2)
drf = operators.D(taurf / 2, D)

adc = operators.ADC
kgrid = 10

# with diffusion
init = [erf, s1, d1, e1]
pre = [s2, d2, e2, sS, dS, eS, erf]  # , drf
post = [erf, sS, dS, eS, s2, d2, e2]  # [drf, ...
seq = [exc, init, trf, post] + [pre, trf, post] * ETL + [adc]
signal = functions.simulate(seq, kgrid=kgrid)[0]

# without diffusion
initn = [erf, s1, e1]
pren = [s2, e2, sS, eS, erf]
postn = [erf, sS, eS, s2, e2]
seqn = [exc, initn, trf, postn] + [pren, trf, postn] * ETL + [adc]
signaln = functions.simulate(seqn, kgrid=kgrid)[0]


# effective b-factor (s / mm^2)
bfactor = -np.log(np.abs(signal[1:] / signaln[1:])) / D  # / np.pi


# plot
fig, axes = plt.subplots(ncols=2, sharex=True, num="diff-sensitivity")
plt.sca(axes.flat[0])
plt.plot(angles, np.abs(signal))
plt.xlabel("angles")
plt.ylabel("intensity (a.u)")
plt.grid()
plt.title("Signal intensity")

plt.sca(axes.flat[1])
plt.plot(angles[1:], bfactor)
plt.xlabel("angles")
plt.ylabel("b-factor (s/mm^2)")
plt.grid()
plt.title("b-factor")
plt.tight_layout()


#
# traps (variable flip angle)

vangles = [120, 85, 70, 62, 60]
vangles += list(np.linspace(60, 180, ETL))
vrf = [operators.T(a, 0) for a in vangles]
seq = [exc, init, vrf[0], post] + [[pre, rf, post, adc] for rf in vrf[1:]]
seqn = [exc, initn, vrf[0], postn] + [[pren, rf, postn, adc] for rf in vrf[1:]]
vsignal = functions.simulate(seq, kgrid=kgrid)
vsignaln = functions.simulate(seqn, kgrid=kgrid)


# effective b-factor (s / mm^2)
vbfactor = -np.log(np.abs(vsignal / vsignaln)).ravel() / D  # / np.pi
vbfactor_rel = np.diff(vbfactor)

# plot
fig, axes = plt.subplots(ncols=3, num="diff-sensitivity-traps")
plt.sca(axes.flat[0])
plt.plot(vangles, "o-")
plt.xlabel("refocussing pulse")
plt.ylabel("flip angles (°)")
plt.ylim(0, 190)
plt.grid()
plt.title("Flip angle")

plt.sca(axes.flat[1])
plt.bar(np.arange(len(vbfactor)) + 1, vbfactor)
plt.xlabel("echo index")
plt.ylabel("b-factor (s/mm^2)")
plt.grid()
plt.title("total b-factor")

plt.sca(axes.flat[2])
plt.bar(np.arange(len(vbfactor_rel)) + 1.5, vbfactor_rel, width=1)
plt.xlabel("echo index")
plt.ylabel("b-factor (s/mm^2)")
plt.grid()
plt.title("b-factor per echo interval")
plt.tight_layout()

plt.show()
