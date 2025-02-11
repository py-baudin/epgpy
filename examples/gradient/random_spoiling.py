"""
Random Spoiling in Fast Gradient Echo Imaging.

  Lin W, Song H
  Improved signal spoiling in fast radial gradient-echo imaging:
  Applied to accurate T 1 mapping and flip angle correction
  Magn Reson Med 2009; 62:1185–1194.

    Here, we use a Bloch simulations with N isochromats to simulate the effects of gradients.
    We could also have used the general Shift operator with non-integer k, to the same results,
    but this takes longer.

    I could not (yet?) get the same results as shown in the article's figures.

"""

import numpy as np
from matplotlib import pyplot as plt
import epgpy as epg

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# parameters
T1 = 60
T2 = 40
Nrf = 400
TR = 1
kgrid = 1e-1

# sequences

# ideal spoiling
print("Ideal spoiling")
FA = [10, 60]
rlx = epg.E(TR, T1, T2)
adc = epg.ADC
rfs = epg.T(FA, 0)
spl = epg.SPOILER
seq = [[rfs, rlx, spl]] * Nrf + [[rfs, adc]]
sim_ideal = epg.simulate(seq)[0]

# rf spoiling
print("quadratic RF spoiling")
FA = [10, 60]
PH = np.arange(0, 181, 3)
rfs = [epg.T(FA, [(n + 1) * n * PH / 2]) for n in range(Nrf)]
spl = epg.S(1)
seq = [[rf, rlx, spl] for rf in rfs[:-1]] + [[rfs[-1], adc]]
sim_rf = epg.simulate(seq, max_nstate=20)[0]

# plot
plt.figure("ideal-vs-rf")
plt.hlines(
    np.abs(sim_ideal[0]),
    0,
    180,
    linestyle=":",
    label=f"Ideal spoiling (FA=10°)",
    color=colors[0],
)
plt.hlines(
    np.abs(sim_ideal[1]),
    0,
    180,
    linestyle=":",
    label=f"Ideal spoiling (FA=60°)",
    color=colors[1],
)
plt.plot(
    PH,
    np.abs(sim_rf)[0],
    "s-",
    label=f"RF-spoiling (FA=10°)",
    color=colors[0],
    ms=3,
    mfc="white",
    zorder=10,
    clip_on=False,
)
plt.plot(
    PH,
    np.abs(sim_rf)[1],
    "o-",
    label=f"RF-spoiling (FA=60°)",
    color=colors[1],
    ms=3,
    mfc="white",
    zorder=10,
    clip_on=False,
)
plt.xticks(np.arange(7) * 30)
plt.xlim(0, 180)
plt.xlabel("RF phase angle (°)")
plt.ylabel("Signal magnitude (a.u)")
plt.title("Ideal vs RF spoiled ")
plt.legend(loc="upper right")
plt.tight_layout()


#
# Random spoiling

#
# profiles and steady state
print("Profiles at end of TR")

random = np.random  # .RandomState(1)

Nrf = 400
TR = 1

npoint = 1001
pos = np.linspace(0, 1, npoint)
freqs = pos - 0.5

# ideal spoiling
rf = epg.T(60, 180)
rlx = epg.E(TR, 60, 40)
spl = epg.SPOILER
seq = [[rf, epg.ADC, rlx, spl]] * Nrf
sim_ideal = epg.simulate(seq)

# operators
Mx = 20
spl = epg.P(1, freqs)
spl20 = epg.P(Mx, freqs)
adc = epg.Adc(weights=1 / npoint)
img = epg.ADC
phq = np.array([(n + 1) * n / 2 * 117 for n in range(Nrf)])
rfq = [epg.T(60, 180 + phi) for phi in phq]
phr = random.uniform(1, 360, Nrf)
rfr = [epg.T(60, 180 + phi) for phi in phr]
krs = random.uniform(0.5, 1, Nrf)
spr = [epg.P(1, k * freqs) for k in krs]
spr20 = [epg.P(Mx, k * freqs) for k in krs]

print("\tquadratic spoiling (phi=117°)")
seq = [[rf, rlx, spl] for rf in rfq] + [img]
sim_rf = epg.simulate(seq, asarray=False)
prof_rf = sim_rf[-1]

seq = [[rf, adc, rlx, spl20] for rf in rfq]
sim_rf20 = epg.simulate(seq, asarray=False)
stdy_rf20 = np.r_[sim_rf20] * np.exp(-1j * np.pi * phq / 180)

print("\trandom rf spoiling")
seq = [[rf, rlx, spl] for rf in rfr] + [img]
sim_rdrf = epg.simulate(seq, asarray=False)
prof_rdrf = sim_rdrf[-1]

seq = [[rf, adc, rlx, spl20] for rf in rfr]
sim_rdrf20 = epg.simulate(seq, asarray=False)
stdy_rdrf20 = np.r_[sim_rdrf20] * np.exp(-1j * np.pi * phr / 180)

print("\trandom gradient spoiling")
seq = [[rf, adc, rlx, sp] for rf, sp in zip(rfq, spr20)]
sim_rdgr20 = epg.simulate(seq, asarray=False)
stdy_rdgr20 = np.r_[sim_rdgr20] * np.exp(-1j * np.pi * phq / 180)

print("\trandom rf and gradient spoiling")
seq = [[rf, rlx, sp] for rf, sp in zip(rfr, spr)] + [img]
sim_rd = epg.simulate(seq, asarray=False)
prof_rd = sim_rd[-1]
seq = [[rf, adc, rlx, sp] for rf, sp in zip(rfr, spr20)] + [img]
sim_rd20 = epg.simulate(seq, asarray=False)
stdy_rd20 = np.r_[sim_rd20[:-1]] * np.exp(-1j * np.pi * phr / 180)
prof_rd20 = sim_rd20[-1]


# approach to steady states
titles = [
    "Quadratic RF spoiling",
    "Random RF spoiling",
    "Random gradient spoiling",
    "Random RF and gradient spoiling",
]
signals = [stdy_rf20, stdy_rdrf20, stdy_rdgr20, stdy_rd20]
ideal = sim_ideal[:, 0]

plt.figure("steady-states", figsize=(8, 6))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.plot(np.abs(signals[i]), label="Signal", color=colors[0], zorder=2)
    plt.plot(np.abs(ideal), "k:", label="Ideal", zorder=3)
    plt.title(titles[i])
    plt.yticks(np.linspace(0, 0.2, 5))
    plt.ylim(0, 0.2)
    plt.ylabel("signal magnitude (a.u)")
    plt.legend(loc="upper right")
    plt.twinx()
    plt.plot(
        np.angle(signals[i]) * 180 / np.pi,
        label="Phase",
        color=colors[1],
        alpha=0.5,
        zorder=1,
    )
    plt.plot(np.angle(ideal) * 180 / np.pi, "k:", alpha=0.5, zorder=3)
    plt.yticks([-180, -90, 0, 90, 180])
    plt.ylabel("Phase (rad)")
plt.xlabel("Rf index")
plt.suptitle("Approach to steady state")
plt.tight_layout()

# profiles
plt.figure("pixel-profiles", figsize=(8, 6))
titles = [
    "Quadratic RF spoiling",
    "Random RF spoiling",
    "Random gradient (1 cycle/pix)",
    "Random gradient (20 cycle/pix)",
]
profiles = [prof_rf, prof_rdrf, prof_rd, prof_rd20]

ax = None
xmin, xmax = pos.min(), pos.max()
for i in range(4):
    ax = plt.subplot(2, 2, i + 1, sharex=ax, sharey=ax)
    plt.plot(pos, np.abs(profiles[i]), color=colors[0])
    plt.plot(
        pos[[0, -1]], np.abs(profiles[i][[0, -1]]), "o", color=colors[0], mfc="white"
    )
    plt.hlines(
        np.abs(np.mean(profiles[i])),
        xmin,
        xmax,
        linestyle=":",
        color=colors[0],
    )
    plt.ylim(0, 0.2)
    if i > 1:
        plt.xlabel("location (pixel)")
    plt.ylabel("signal magnitude (a.u)")
    plt.title(titles[i])
    plt.twinx()
    plt.plot(pos, np.angle(profiles[i]) * 180 / np.pi, color=colors[1], alpha=0.5)
    plt.yticks([-180, -90, 0, 90, 180])
    plt.ylabel("Phase (rad.)")

plt.suptitle("Pixel profiles at end of TR")
plt.tight_layout()


#
# signal ratios
NAX = np.newaxis

TR = 1  # ms
T1, T2 = [60, 200], 40
N1, N2 = 4 * 200 // TR, 400
Mx = [2, 5, 10, 20, 50, 100]  # max gradient moments (cycles/pix)

npoint = 360
pos = np.linspace(0, 1, npoint)
freqs = pos - 0.5

# operators
adc = epg.ADC
rf = epg.T(60, 0)
sp = epg.SPOILER
rlx = epg.E(TR, T1, T2)
phq = np.array([(n + 1) * n * 117 / 2 for n in range(N1 + N2)])
rfq = [epg.T(60, phi) for phi in phq]
phr = np.random.uniform(0, 360, N1 + N2)
rfr = [epg.T(60, phi) for phi in phr]
spr = [epg.P([Mx], k * freqs[NAX, NAX]) for k in np.random.uniform(0.5, 1, N1 + N2)]

print("Ideal spoiling")
seq = [[rf, rlx, sp]] * N1 + [[rf, adc, rlx, sp]] * N2
sim_ideal = epg.simulate(seq)
sim_ideal = sim_ideal[-1]

print(f"random gradient spoiling")
seq = [[rf, rlx, sp] for rf, sp in zip(rfq[:N1], spr[:N1])]
seq += [[rf, adc, rlx, sp] for rf, sp in zip(rfq[N1:], spr[N1:])]
sim_rdgr = epg.simulate(seq).sum(-1) / npoint
sim_rdgr *= np.exp(-1j * np.pi * phq[N1:] / 180)[:, NAX, NAX]

print(f"random rf and gradient spoiling")
seq = [[rf, rlx, sp] for rf, sp in zip(rfr[:N1], spr[:N1])]
seq += [[rf, adc, rlx, sp] for rf, sp in zip(rfr[N1:], spr[N1:])]
sim_rd = epg.simulate(seq).sum(-1) / npoint
sim_rd *= np.exp(-1j * np.pi * phr[N1:] / 180)[:, NAX, NAX]


def nmean(sig, ref):
    return np.mean(np.abs(sig), axis=0) / np.abs(ref[:, NAX])


def nstd(sig, ref):
    return np.std(np.abs(sig), axis=0) / np.abs(ref[:, NAX])


#
# plot
plt.figure("signal-ratios")
ax = plt.subplot(1, 2, 1)
h = plt.plot(Mx, nmean(sim_rdgr, sim_ideal).T, "s-")
plt.hlines(1, Mx[0], Mx[-1], color="k", linestyle=":")
plt.title("Random gradient spoiling")
plt.xticks([0, 20, 40, 60, 80, 100])
plt.legend(h, [f"T1/TR={t1}" for t1 in T1], loc="upper right")
plt.gca().set_prop_cycle(None)
plt.plot(Mx, nstd(sim_rdgr, sim_ideal).T, "s:")
plt.ylim(0, 2)

ax = plt.subplot(1, 2, 2)
h = plt.plot(Mx, nmean(sim_rd, sim_ideal).T, "s-")
plt.hlines(1, Mx[0], Mx[-1], color="k", linestyle=":")
plt.title("Random gradient and RF spoiling")
plt.xticks([0, 20, 40, 60, 80, 100])
plt.legend(h, [f"T1/TR={t1}" for t1 in T1], loc="upper right")
plt.gca().set_prop_cycle(None)
plt.plot(Mx, nstd(sim_rd, sim_ideal).T, "s:")
plt.ylim(0, 2)

plt.tight_layout()
plt.show()
