"""
Simulation of MRF sensitivity to off-resonance patterns

Based on:
> Gao X, Kiselev VG, Lange T, Hennig J, Zaitsev M:
  Three‐dimensional spatially resolved phase graph framework. Magn Reson Med 2021; 86:551–560.
  Part "3.1 Off-resonance simulation"

"""

import time
import numpy as np
from matplotlib import pyplot as plt
from epgpy import operators, functions, utils

NAX = np.newaxis

gamma = utils.gamma_1H  # Hz/mT
FOV = 0.128  # m

# static off-resonance gradient
Freq = 100  # Hz
G = Freq / (FOV / 2) / gamma  # mT/m
pos = np.linspace(-0.5, 0.5, 501) * FOV  # m
offres = utils.space_to_freq(G, pos * 1e3)  # kHz
wfreqs = offres * 2 * np.pi * 1e3  # rad/s

# sequence
Nrf = 100  # number of rfs
# pSSFP sequence of paramters
FA0 = (
    10
    + np.sin(2 * np.pi * np.linspace(1, 250, Nrf) * 1e-3) * 50
    + np.random.uniform(-8.66, 8.66, Nrf)
)
FA, TE, TR = [FA0[0] / 2], [0], []
TRssfp = 10
for i in range(1, Nrf):
    # alpha_i
    fa = FA0[i] / 2 + FA0[i - 1] / 2

    # c_i
    c = np.sin(FA0[i - 1] / 2 * np.pi / 180) / np.sin(FA0[i] / 2 * np.pi / 180)

    if c < 1:
        tr = TRssfp / 2 + TE[-1]
        te = (tr - TE[-1]) * c
    else:
        te = TRssfp / 2
        tr = te / c + TE[-1]

    FA.append(fa)
    TE.append(te)
    TR.append(tr)

TR.append(TRssfp)

k1 = [utils.get_wavenumber(G, TE[i]) for i in range(Nrf)]
k2 = [utils.get_wavenumber(G, TR[i] - TE[i]) for i in range(Nrf)]

# tissue
T1 = 1084  # ms
T2 = 68  # ms


# sequence
null = operators.NULL
adc = operators.ADC
rx1 = [null if i == 0 else operators.E(TE[i], T1, T2) for i in range(Nrf)]
rx2 = [operators.E(TR[i] - TE[i], T1, T2) for i in range(Nrf)]
g1 = [null if i == 0 else operators.S(k1[i]) for i in range(Nrf)]
g2 = [operators.S(k2[i]) for i in range(Nrf)]
rf = [operators.T(FA[i], 180 * (i % 2)) for i in range(Nrf)]

seq = [[rf[i], g1[i], rx1[i], adc, g2[i], rx2[i]] for i in range(Nrf)]

# reference (Bloch sim)
# add offset frequency
rxg1 = [null if i == 0 else operators.E(TE[i], T1, T2, g=offres) for i in range(Nrf)]
rxg2 = [operators.E(TR[i] - TE[i], T1, T2, g=offres) for i in range(Nrf)]
seqr = [[rf[i], rxg1[i], adc, rxg2[i]] for i in range(Nrf)]
sigr = functions.simulate(seqr)[-1]


#
# simulate

# initial gridsize
Kg = 20  # rad/m
# reduction factor
fk = 0.2

sims = {}
stats = {}
print(f"iteration 0: Kg={Kg:.2f}")
for iter in range(100):
    tic = time.time()
    Fs, ks = functions.simulate(seq, kgrid=Kg, probe=("F", "k"), asarray=False)
    duration = time.time() - tic
    # discrete Fourier transform
    sig = functions.dft(pos[..., NAX], Fs[-1], ks[-1])
    # store signal and update Kg
    stats[Kg] = {"time": duration, "num-k": [k.size for k in ks]}
    sims[Kg] = sig
    if iter:
        diff = np.linalg.norm(sig - sims[prevK]) / np.linalg.norm(sig)
        print(f"iteration {iter}: Kg={Kg:.2f}, diff={diff:.3f}")
        if diff < 1e-2:
            break
    prevK = Kg
    Kg *= fk

print("done")

#
# plot

fig, axes = plt.subplots(nrows=2, ncols=2, num="pssfp", figsize=(12, 7))
index = np.arange(Nrf) + 1

# pSSFP spectrum
plt.sca(axes[1, 1])
for Kg in sorted(sims)[::-1]:
    plt.plot(wfreqs.ravel(), np.abs(sims[Kg]).ravel(), label=f"Kg={Kg:0.2f} rad/s")
plt.plot(wfreqs.ravel(), np.abs(sigr).ravel(), "k:", label=f"Reference (Bloch)")
plt.xlabel("Off-resonance frequency (rad/s)")
plt.ylabel("Magnitude (a.u)")
plt.title("Spectrum of pSSFP")
plt.legend(loc="upper right")

# burden
plt.sca(axes[0, 1])
for Kg in sorted(stats)[::-1]:
    plt.plot(index, stats[Kg]["num-k"], label=f"Kg={Kg:0.2f} rad/s")
plt.gca().set_yscale("log")
ylim = plt.ylim()
plt.plot(
    index, np.exp(2 * (index - 1) + 1) / np.exp(1), "k:", label="Exponential function"
)
plt.ylim(*ylim)
plt.xlabel("RF pulse index")
plt.ylabel("Num of k vectors")
plt.legend(loc="lower right")
plt.title("Burden")

# efficiency
plt.sca(axes[1, 0])
kvals = sorted(stats)[::-1]
tvals = [stats[k]["time"] for k in kvals]
errors = [100 * np.linalg.norm(sigr - sims[k]) / np.linalg.norm(sigr) for k in kvals]
h1 = plt.plot(kvals, tvals, "-o", label="Computation time")
plt.ylabel("Time (s)")
plt.xlabel("K-grid radius (rad/m)")
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
plt.xticks(kvals, [f"{k:.2f}" for k in kvals])
plt.xlim(plt.xlim()[::-1])
plt.twinx()
h2 = plt.plot(kvals, errors, "-o", color="orange", label="Residual Error")
plt.ylabel("Error (%)")
plt.legend(handles=h1 + h2, loc="upper center")
plt.title("Efficiency")

# sequence parameters
plt.sca(axes[0, 0])
h1 = plt.plot(index, TE, color="blue", label="TE")
h2 = plt.plot(index, TR, color="orange", label="TR")
plt.ylabel("Time (ms)")
plt.xlabel("RF pulse index")
plt.twinx()
h3 = plt.plot(np.arange(Nrf) + 1, FA, color="green", label="FA")
plt.ylabel("Flip angle (degree)")
plt.legend(handles=[h1[0], h2[0], h3[0]])
plt.title("Protocol parameters")


plt.suptitle(
    "Simulation of off-resonance effects in pSSFP-MRF with spatially-resolved EPG"
)
plt.tight_layout()
plt.show()
