""" 1d inverse Laplace Transform

An attempt at implementing similar examples as found in:

> Eads CD: Analysis of multicomponent exponential magnetic resonance relaxation data: 
  automatable parameterization and interpretable display protocols.

Does not work terribly well for now...

"""

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from epgpy.utilities.ilt1d import *

NAX = np.newaxis


#
# test signals
npoint = 500  # 50
times = np.linspace(0, 50, npoint)  # ms
Ta, Tb = 5, 20
siga = np.exp(-times / Ta)
sigb = np.exp(-times / Tb)

# multi-comp signal
ndist = 10
Adist = np.r_[[0], np.random.uniform(0, 1, ndist), [0]]
boundsc = [1 / 50, 1]
Rdist = np.geomspace(boundsc[0], boundsc[1], len(Adist))
Rc = np.geomspace(1 / 100, 1, 100)
bspline = interpolate.make_interp_spline(Rdist, Adist, k=3, bc_type="natural")
Ac = bspline(Rc)
Ac[(Rc > boundsc[1]) | (Rc < boundsc[0]) | (Ac < 0)] = 0
sigc = (Ac * np.exp(-np.outer(times, Rc))).mean(axis=1)
sigc /= sigc[0]

names = ["a", "b", "0.3a + 0.7b", "c"]
signals = [siga, sigb, 0.3 * siga + 0.7 * sigb, sigc]
gt = [
    {"r": 1 / Ta, "a": 1},
    {"r": 1 / Tb, "a": 1},
    {"r": [1 / Ta, 1 / Tb], "a": [0.3, 0.7]},
    {"r": Rc, "a": Ac},
]

# add noise
sigma = 1e-2
noise = sigma * np.random.normal(size=npoint)
signals = [sig + noise for sig in signals]
# signals = [np.abs(sig + noise) for sig in signals]

# resolution
bounds = get_bounds(times)
res, num = get_resolution(times, bounds)
rates, kernel = get_kernel(times, bounds, num)
print(f"Bounds: {bounds}, res: {res}")

# ITL
nsignal = len(signals)
arm = []
armls = []
for i in range(nsignal):
    print(f"signal {names[i]}")
    # ARM
    r, a = ilt1d(times, signals[i], kernel=kernel, ls=False)
    ir, ia = ilt1d_crb(times, signals[i], r, a)
    rc, ac = qcr(bounds, r, a, ir)
    arm.append({"r": r, "a": a, "rc": rc, "ac": ac, "ir": ir, "ia": ia})

    # ARM + LS
    r, a = ilt1d(times, signals[i], kernel=kernel, ls=True)
    ir, ia = ilt1d_crb(times, signals[i], r, a)
    rc, ac = qcr(bounds, r, a, ir)
    armls.append({"r": r, "a": a, "rc": rc, "ac": ac, "ir": ir, "ia": ia})

# plot prediction and ILT
plt.figure(figsize=(10, 8), num="ilt")

logres = np.log(res)
colors = plt.cm.plasma(np.linspace(0, 1, nsignal + 1))

plt.subplot(2, 1, 1)
for i in range(nsignal - 1):
    name, color = names[i], colors[i]
    plt.plot(times, signals[i], color=color, label=f"{name} (signal)")
    plt.plot(
        times,
        flt1d(times, arm[i]["r"], arm[i]["a"]),
        "s",
        alpha=0.5,
        color=color,
        label=f"{name} (ARM)",
    )
    plt.plot(
        times,
        flt1d(times, armls[i]["r"], armls[i]["a"]),
        "o",
        alpha=0.5,
        color=color,
        label=f"{name} (ARM+LS)",
    )
plt.legend(loc="upper right")
plt.ylabel("signal (a.u)")
plt.xlabel("times (ms)")
plt.title(f"Signal vs prediction (resolution={res:.2f})")

# plot ARM fit
plt.subplot(2, 2, 3)
for i in range(nsignal - 1):
    name, color = names[i], colors[i]
    fit = arm[i]
    h = plt.stem(np.log(fit["r"]), fit["a"], label=f"{name}")
    plt.setp(h.stemlines, color=color, alpha=0.5)
    plt.setp(h.markerline, color=color)
    # uncertainties
    plt.bar(np.log(fit["r"]), height=fit["ia"], width=fit["ir"], color=color, alpha=0.3)
    # continuous render
    plt.plot(fit["rc"], fit["ac"], color=color)
    # gt
    plt.scatter(np.log(gt[i]["r"]), gt[i]["a"], marker="+", color=color)
    plt.scatter(np.log(rates), [0] * len(rates), marker="o", color="k")
plt.xlabel("$log(R)$")
plt.ylabel("ILT (a.u)")
plt.title(f"ILT (ARM)")
plt.grid()
plt.legend(loc="upper right")

plt.subplot(2, 2, 4)
for i in range(nsignal - 1):
    name, color = names[i], colors[i]
    fit = armls[i]
    h = plt.stem(np.log(fit["r"]), fit["a"], label=f"{name}")
    plt.setp(h.stemlines, color=color, alpha=0.5)
    plt.setp(h.markerline, color=color)
    plt.plot(fit["rc"], fit["ac"], color=color)
    # gt
    plt.scatter(np.log(gt[i]["r"]), gt[i]["a"], marker="+", color=color)
    plt.scatter(np.log(rates), [0] * len(rates), marker="o", color="k")
plt.xlabel("$log(R)$")
plt.ylabel("ILT (a.u)")
plt.title(f"ILT (ARM+LS)")
plt.grid()
plt.legend(loc="upper right")

plt.suptitle(f"ILT for mixed signals (sigma={sigma:0.1e})")
plt.tight_layout()

#
# multi-comp
color = "b"
name = "signal c"

plt.figure(figsize=(10, 8), num="multi-comp")

plt.subplot(1, 2, 1)
plt.plot(times, sigc, "k:", label="Ground truth")
plt.plot(
    times,
    flt1d(times, arm[-1]["r"], arm[-1]["a"]),
    "s",
    alpha=0.5,
    color=color,
    label=f"{name} (ARM)",
)
plt.plot(
    times,
    flt1d(times, armls[-1]["r"], armls[-1]["a"]),
    "o",
    alpha=0.5,
    color=color,
    label=f"{name} (ARM+LS)",
)
plt.legend(loc="upper right")
plt.ylabel("Signal (a.u)")
plt.ylabel("times (ms)")
plt.title("Signal and prediction")

# plot ARM-LS fit
plt.subplot(1, 2, 2)
plt.plot(np.log(Rc), Ac, "k:", label="Ground truth")
h = plt.stem(np.log(armls[-1]["r"]), armls[-1]["a"], label=f"ARM")
plt.setp(h.stemlines, color=color, alpha=0.5)
plt.setp(h.markerline, color=color)
# uncertainties
plt.bar(
    np.log(armls[-1]["r"]),
    height=arm[-1]["ia"],
    width=armls[-1]["ir"],
    color=color,
    alpha=0.3,
)
# continuous render
plt.plot(armls[-1]["rc"], armls[-1]["ac"], color=color)

plt.xlabel("$log(R)$")
plt.ylabel("ILT (a.u)")
plt.grid()
plt.title("ARM-LS fit")
plt.suptitle("Multi-component signal")
plt.tight_layout()

#
# resolution
print("Get resolution for various tolerance values")
tols = np.logspace(-5, -1, 20)
resolutions = [get_resolution(times, bounds, tol=tol)[0] for tol in tols]

fig = plt.figure("resolution", figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(tols, resolutions)
plt.xscale("log")
plt.xlabel("Maximum tolerable deviation")
plt.ylabel("Geometric increment")
plt.title("Resolution vs max. deviation")

# kernel
plt.subplot(1, 2, 2)
h = plt.plot(times, kernel)
labels = [f"{r:.2}$ms^{{-1}}$" for r in rates]
plt.legend(labels=labels, loc="upper left", bbox_to_anchor=(1, 1.02))
plt.title(
    f"Basis set (n={len(labels)}, resolution={res:.2f}, bounds={bounds[0]:.2}-{bounds[1]:.2}$ms^{{-1}}$)"
)
plt.xlabel("times (ms)")
plt.ylabel("signal (a.u)")

plt.tight_layout()

plt.show()
