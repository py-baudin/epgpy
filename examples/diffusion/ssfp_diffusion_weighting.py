"""
Diffusion-weighted SSFP simulation

based on:
> Gao X, Kiselev VG, Lange T, Hennig J, Zaitsev M: 
  Three‐dimensional spatially resolved phase graph framework. Magn Reson Med 2021; 86:551–560.
  Part "3.1 Off-resonance simulation"

"""
import numpy as np
from matplotlib import pyplot as plt
from epgpy import operators, functions, utils

NAX = np.newaxis
gamma = utils.gamma_1H  # Hz/mT

#
# sequence parameters
Nrf = 100
FA = 25  # degree
Gdiff = 23.5  # mT/m
Tdiff = 5  # ms
TR = 10  # ms

# rotation matrix
angle = 45 * np.pi / 180  # rad
rmat = np.array(
    [
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ]
)

#
# tissue
T1 = 1084  # ms
T2 = 68  # ms
D = np.diag([1.35, 0.5, 0]) * 1e-3  # mm^2/s

# static gradient
FOV = 0.128  # m
Freq = 100  # Hz (frequency offset)
G = Freq / (FOV / 2) / gamma  # mT/m
pos = np.c_[np.zeros((501, 2)), np.linspace(-0.5, 0.5, 501) * FOV]  # m
offres = utils.space_to_freq(G, pos[:, 2] * 1e3)  # kHz
wfreqs = offres * 2 * np.pi * 1e3  # rad/s


#
# simulation
# adc = operators.ADC
adc = operators.Imaging(pos)
# rf1 pulse
rf1 = operators.T(FA, 0)
rf2 = operators.T(FA, 180)
# gradient
gradx = [Gdiff, 0, G]
grady = [0, Gdiff, G]
g1x = operators.G(Tdiff, gradx)
g1y = operators.G(Tdiff, grady)
g2 = operators.G(TR - Tdiff, [0, 0, G])
# diffusion
d1x = operators.D(Tdiff, D, g1x.k)
d1y = operators.D(Tdiff, D, g1y.k)
d2 = operators.D(TR - Tdiff, D, g2.k)
# relax
rx1 = operators.E(Tdiff, T1, T2)
rx2 = operators.E(TR - Tdiff, T1, T2)

# sequence
seq0 = (Nrf // 2) * [
    [rf1, [g1x, d1x, rx1], [g2, d2, rx2], adc],
    [rf2, [g1x, d1x, rx1], [g2, d2, rx2], adc],
]
seq0qi = (Nrf // 2) * [
    [rf1, [g1x, d1x, rx1], [g2, d2, rx2], adc],
    [rf2, [g1y, d1y, rx1], [g2, d2, rx2], adc],
]

# seq2 45
g1x = operators.G(Tdiff, rmat @ gradx)
g1y = operators.G(Tdiff, rmat @ grady)
d1x = operators.D(Tdiff, D, g1x.k)
d1y = operators.D(Tdiff, D, g1y.k)

seq45 = (Nrf // 2) * [
    [rf1, [g1x, d1x, rx1], [g2, d2, rx2], adc],
    [rf2, [g1x, d1x, rx1], [g2, d2, rx2], adc],
]
seq45qi = (Nrf // 2) * [
    [rf1, [g1x, d1x, rx1], [g2, d2, rx2], adc],
    [rf2, [g1y, d1y, rx1], [g2, d2, rx2], adc],
]

# grid size
Kg = 0.1  # rad/m
sig0 = functions.simulate(seq0, kgrid=Kg)
sig45 = functions.simulate(seq45, kgrid=Kg)
sig0qi = functions.simulate(seq0qi, kgrid=Kg)
sig45qi = functions.simulate(seq45qi, kgrid=Kg)


# Bloch-sim for background gradient
# s = operators.S(1)
# rx1 = operators.E(Tdiff, T1, T2, g=offres)
# rx2 = operators.E(TR - Tdiff, T1, T2, g=offres)
# seq = [[rf1, rx1, rx2, adc]] * Nrf
# sig_bloch = functions.simulate(seq)


#
# plot
fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 6))
plt.sca(axes.flat[0])
plt.plot(wfreqs, np.abs(sig0[48]).ravel(), label="conventional (0°)")
plt.plot(wfreqs, np.abs(sig0qi[48]).ravel(), "-.", label="quasi-isotropic (0°)")
plt.plot(wfreqs, np.abs(sig45[48]).ravel(), label="conventional (45°)")
plt.plot(wfreqs, np.abs(sig45qi[48]).ravel(), "-.", label="quasi-isotropic (45°)")
plt.title("Echo #49")
plt.ylabel("Magnitude (a.u)")
plt.legend(loc="lower right")

plt.sca(axes.flat[1])
plt.plot(wfreqs, np.abs(sig0[49]).ravel(), label="conventional (0°)")
plt.plot(wfreqs, np.abs(sig0qi[49]).ravel(), "-.", label="quasi-isotropic (0°)")
plt.plot(wfreqs, np.abs(sig45[49]).ravel(), label="conventional (45°)")
plt.plot(wfreqs, np.abs(sig45qi[49]).ravel(), "-.", label="quasi-isotropic (45°)")
plt.title("Echo #50")
plt.ylabel("Magnitude (a.u)")
plt.legend(loc="lower right")

plt.ylim(0, plt.ylim()[1])
plt.xlabel("Off-resonance frequency (rad/s)")

plt.suptitle("Frequency spectra of Steady-State DWI sequences")
plt.tight_layout()
plt.show()
