"""
Point-resolved spectroscopy simulation

based on:
> Gao X, Kiselev VG, Lange T, Hennig J, Zaitsev M: 
  Three‐dimensional spatially resolved phase graph framework. Magn Reson Med 2021; 86:551–560.
  Part "3.3 Point-resolved spectroscopy simulation"
"""
import numpy as np
from matplotlib import pyplot as plt
from epgpy import operators, functions, utils

# constants
gamma = utils.gamma_1H  # Hz/mT

# FOV
FOV = 48  # mm
npix = 16
grid = (
    FOV
    * 1e-3
    * np.stack(np.meshgrid(*[np.linspace(-0.5, 0.5, npix)] * 3, indexing="ij"), axis=-1)
)

# filter k space (?)
kfilt = 2 * np.pi / (FOV * 1e-3 / npix)  # rad/m

# sequence
FA = 90  # degree
TE = 30  # ms
TE1, TE2 = 14, 16  # ms
Kg = 1  # rad/m

# spoiler gradients
kc = 2 * np.pi * 500  # rad/m

# static gradients
Gs = np.array([0.1, -0.2, 0.3]) / gamma * 1e2  # mT/m

# imaging gradients (rad/m)
kim = 2 * np.pi * npix / FOV * 1e3

# PRESS sequence
rf1 = operators.T(90, 90)
rf2 = operators.T(90, 0)
rf3 = operators.T(90, 0)
eye = 0.5 * np.eye(3)
gx, gy, gz = map(operators.S, [eye[0] * kim, eye[1] * kim, eye[2] * kim])
gc = operators.S([kc] * 3)
gs1 = operators.G(TE1 / 2, Gs, duration=True)
gs2 = operators.G(TE2 / 2, Gs, duration=True)
gs500 = operators.G(500, Gs, duration=True)

adc = operators.ADC
seq = [
    [rf1],
    [gs1, gc, gy, rf2, gy, gc, gs1],
    adc,
    [gs2, gc, gz, rf3, gz, gc, gs2],
    adc,
    [gs500],
    adc,
    [gs500],
    adc,
]

times = functions.get_adc_times(seq)
Fs, ks = functions.simulate(seq, kgrid=Kg, probe=("F", "k"), asarray=False)
keep = [np.all(np.abs(ks[i]) <= kfilt, axis=-1) for i in range(4)]  # filter ks
sig = [functions.dft(grid, Fs[i][keep[i]], ks[i][keep[i]]) for i in range(4)]


# DOTCOPS
gc2 = operators.S([kc, kc, 0])
gc3 = gc2
gc4 = operators.S([kc, kc, -kc])
seq = [
    [rf1],
    [gs1, gc, gy, rf2, gy, gc2, gs1],
    adc,
    [gs2, gc3, gz, rf3, gz, gc4, gs2],
    adc,
    [gs500],
    adc,
    [gs500],
    adc,
]
Fs, ks = functions.simulate(seq, kgrid=Kg, probe=("F", "k"), asarray=False)
keep = [np.all(np.abs(ks[i]) <= kfilt, axis=-1) for i in range(4)]
sig_2 = [functions.dft(grid, Fs[i][keep[i]], ks[i][keep[i]]) for i in range(4)]


# plot
def plot_imgs(title, sig, which):
    fignum = title.lower().replace(" ", "_")
    _, axes = plt.subplots(nrows=3, ncols=4, num=fignum, figsize=(8, 6))
    opts = {"interpolation": "nearest", "cmap": "gray"}
    if which == "magnitude":
        func = np.abs
        opts.update({"vmin": 0, "vmax": 0.5})
    if which == "phase":
        func = np.angle
        opts.update({"vmin": -np.pi, "vmax": np.pi})
    ticks = [0, npix / 2, npix - 1], [-FOV / 2, 0, FOV / 2]
    for i in range(4):
        plt.sca(axes[0, i])
        plt.imshow(func(sig[i][..., 10]), **opts)
        if i == 0:
            plt.ylabel("X (mm)")
            plt.xlabel("Y (mm)")
        plt.xticks(*ticks)
        plt.yticks(*ticks)
        plt.title(f"Time {i + 1}")

        plt.sca(axes[1, i])
        plt.imshow(func(sig[i][:, 10]), **opts)
        if i == 0:
            plt.ylabel("X (mm)")
            plt.xlabel("Z (mm)")
        plt.xticks(*ticks)
        plt.yticks(*ticks)

        plt.sca(axes[2, i])
        plt.imshow(func(sig[i][10]), **opts)
        if i == 0:
            plt.ylabel("Y (mm)")
            plt.xlabel("Z (mm)")
        plt.xticks(*ticks)
        plt.yticks(*ticks)

    plt.suptitle(title)
    plt.tight_layout()


plot_imgs("Evolution of magnetization for PRESS (magnitude)", sig, "magnitude")
plot_imgs("Evolution of magnetization for PRESS (phase)", sig, "phase")
plot_imgs("Evolution of magnetization for DOTCOPS (magnitude)", sig_2, "magnitude")
plot_imgs("Evolution of magnetization for DOTCOPS (phase)", sig_2, "phase")


plt.show()
