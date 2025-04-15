"""Pulse profile simulation"""

import numpy as np
from epgpy import operators, functions, rfpulse
from matplotlib import pyplot as plt

# sinc pulse
npoint = 100
nlobe = 5
pulse = np.sinc(nlobe * np.linspace(-1, 1, npoint))

# parameters
BW = 2  # kHz
duration = nlobe / BW * 2  # ms
FA = 90
T1, T2 = 1e3, 1e2
offres = np.linspace(-3, 3, 301)  # -3 to 3 Hz

# operators
rf = rfpulse.RFPulse(pulse, duration, alpha=90)
# insert off-resonance values (and relaxation)
rf_ = functions.modify(rf, T1=T1, T2=T2, g=offres)

adc = operators.ADC
rewind = operators.P(duration / 2, -offres)
sim = functions.simulate([rf_, rewind, adc])


# plot pulse
fig, axes = plt.subplots(nrows=2, num="sinc-pulse")
plt.sca(axes[0])
time = np.linspace(0, duration, npoint)
plt.plot(time, pulse)
plt.grid(axis="y")
plt.xlabel("time (ms)")
plt.ylabel("Pulse values (a.u)")
plt.title(f"sinc-pulse with BW={BW}kHz, FA={FA}°")


# plot frequency profile
plt.sca(axes[1])
mag = plt.plot(offres, np.abs(sim[0]), label="magnitude")
plt.ylabel("Magnitude (a.u)")
plt.xlabel("Frequency (kHz)")
plt.grid(axis="x")
plt.twinx()
pha = plt.plot(offres, np.angle(sim[0]), "r:", label="phase")
plt.ylabel("Phase (rad)")
plt.ylim(-np.pi, np.pi)
plt.legend(handles=mag + pha, loc="upper right")
plt.title("Pulse frequency profile")

plt.tight_layout()


# use shift operator instead
FOV = 1e-2  # m
# dephasing at each small excitation
kvalue = 2 * np.pi * offres[-1] / (FOV / 2) * duration / npoint  # rad/m

shift = operators.S(1)
rlx = operators.E(duration / npoint, T1, T2)
rewind = operators.S(-npoint // 2)
seq = [[t, rlx, shift] for t in rf.operators]

pos = FOV * np.linspace(-0.5, 0.5, 301)
adc = operators.DFT(pos)
sim2 = functions.simulate(seq + [rewind, adc], kvalue=kvalue)
sim2 = sim2[0]

plt.figure()
plt.plot(offres, np.abs(sim[0]), label="With E operator")
plt.plot(offres, np.abs(sim2[0]), "--", label="With S operator")
plt.ylabel("Magnitude (a.u)")
plt.xlabel("Frequency (kHz)")
plt.legend(loc="upper left")
plt.grid(axis="x")
plt.twinx()
plt.plot(offres, np.angle(sim[0]), "g:", label="With E operator")
plt.plot(offres, np.angle(sim2[0]), "y:", label="With S operator")
plt.ylabel("Phase (rad)")
plt.title("2 approaches for pulse profile simulation")
plt.tight_layout()
plt.show()
