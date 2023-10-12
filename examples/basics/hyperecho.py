"""
Hyperecho simulation

Weigel M
Dephasing, RF pulses, and echoes - pure and simple: Extended Phase Graphs. 
J Magn Reson Imaging 2015; 41:266–295.

"""
import numpy as np
from matplotlib import pyplot as plt
from epgpy import operators, functions, statematrix

# sequence
alpha = 10

Nrf = 223
rf1 = operators.T(alpha, 0)
rf2 = operators.T(-alpha, 0)
exc = operators.T(90, 90)
rfc = operators.T(180, 0)
grad = operators.S(1)
adc = operators.ADC

n = Nrf // 2
seq1 = [exc, grad] + [[rf1, grad, adc]] * n + [rfc, grad] + [[rf2, grad, adc]] * n

# simulate
nmax1 = Nrf // 2 + 10
init = statematrix.StateMatrix(nstate=nmax1, max_nstate=nmax1)
F1, Z1 = functions.simulate(seq1, probe=("F", "Z"), init=init)
F1, Z1 = F1[:, 0].T, Z1[:, 0].T

# varying echo times and phases
alphas = np.round(np.linspace(10, 30, 50), -1)
phis = np.linspace(0, 30, 50)
rf1s = [operators.T(a, p) for a, p in zip(alphas, phis)]
rf2s = [operators.T(-a, -p) for a, p in zip(alphas[::-1], phis[::-1])]
seq2 = [exc, grad] + [[rf, grad, adc] for rf in rf1s]
seq2 += [rfc, grad] + [[rf, grad, adc] for rf in rf2s]
nmax2 = 51
init = statematrix.StateMatrix(nstate=nmax2, max_nstate=nmax2)
F2, Z2 = functions.simulate(seq2, probe=("F", "Z"), init=init)
F2, Z2 = F2[:, 0].T, Z2[:, 0].T


#
# plot
cm = plt.cm.jet.with_extremes(under="k")
fmin = 1e-2
zmin = 1e-3
aspect1 = F1.shape[1] / F1.shape[0]
aspect2 = F2.shape[1] / F2.shape[0]
opts = {"origin": "lower", "cmap": cm, "interpolation": "nearest"}

_, axes = plt.subplots(nrows=2, ncols=2, num="hyperecho", figsize=(9, 7))
plt.sca(axes[0, 0])
plt.imshow(np.abs(F1), vmin=fmin, aspect=aspect1, **opts)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xlabel("# Echos")
plt.yticks([0, nmax1, 2 * nmax1 + 1], ["$|F_{-k}|$", "$|F_0|$", "$|F_k$|"])
plt.title("Transverse (F) states (sequence 1)")

plt.sca(axes[0, 1])
plt.imshow(np.abs(Z1), vmin=zmin, aspect=aspect1, **opts)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xlabel("# Echos")
plt.title("Longitudinal (Z) states(sequence 1)")
plt.yticks([0, nmax1, 2 * nmax1 + 1], ["$|Z_{-k}|$", "$|Z_0|$", "$|Z_k$|"])

plt.sca(axes[1, 0])
plt.imshow(np.abs(F2), vmin=fmin, aspect=aspect2, **opts)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xlabel("# Echos")
plt.title("Transverse (F) states (sequence 2)")
plt.yticks([0, nmax2, 2 * nmax2 + 1], ["$|F_{-k}|$", "$|F_0|$", "$|F_k$|"])

plt.sca(axes[1, 1])
plt.imshow(np.abs(Z2), vmin=zmin, aspect=aspect2, **opts)
plt.colorbar(fraction=0.046, pad=0.04)
plt.xlabel("# Echos")
plt.title("Longitudinal (Z) states (sequence 2)")
plt.yticks([0, nmax2, 2 * nmax2 + 1], ["$|Z_{-k}|$", "$|Z_0|$", "$|Z_k$|"])

plt.suptitle("Hyperecho sequences: state evolution matrices")
plt.tight_layout()

plt.show()
