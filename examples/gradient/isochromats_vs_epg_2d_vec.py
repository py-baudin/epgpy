""" vectorized version
=> much slower due to the DFT in imaging
"""

import pathlib
import time
import numpy as np
from scipy import ndimage
import epgpy as epg
from matplotlib import pyplot as plt

NAX = np.newaxis
random = np.random  # .RandomState(0)

# brain phantom
"""
Source:
    Colin 27 Average Brain 2008
    Copyright (C) 1993–2009 Louis Collins, McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University.
    https://nist.mni.mcgill.ca/category/atlas/
"""
HERE = pathlib.Path(__file__).parent
wm, gm, csf = np.load(HERE / "brain.npy")

# reshape to 64x64
zoom = 64 / wm.shape[0]
wm = ndimage.zoom(wm, zoom)
gm = ndimage.zoom(gm, zoom)
csf = ndimage.zoom(csf, zoom)

mask = np.max([wm, gm, csf], axis=0) > 1e-5

# acquisition
FA = 30  # degrees
TR = 10  # ms
FOV = 200e-3  # m
nread, nphase = mask.shape

pixsize = FOV / nread
pixels = np.mgrid[-nread // 2 : nread // 2, -nphase // 2 : nphase // 2]
pixels = pixels.reshape(2, -1).T[mask.flat] * FOV / [nread, nphase]
# pixels = [np.arange(-nread //2, nread//2) * FOV / nread, np.arange(-nphase //2, nphase//2) * FOV / nphase]

# system
# GM, WM, CSF
PD = [0.8, 0.7, 1.0]  # a.u.
T1 = [1.55e3, 0.83e3, 4.16e3]  # ms
T2 = [0.09e3, 0.07e3, 1.65e3]  # ms
T2p = [0.322e3, 0.183e3, 0.0591e3]  # ms
pds = np.stack([gm * PD[0], wm * PD[1], csf * PD[2]]).reshape(3, -1)[..., mask.flat]

# EPG
print("EPG")
# set proton density and T2* decay
init = epg.System(weights=pds[NAX], modulation=-1 / np.array([T2p]))
# rf pulse and ADC with phase offset
rf = [epg.T(FA, 0)] * nphase
# rf = [epg.T(FA, 117 * i * (i + 1)/2) for i in range(nphase)] # rf spoiling
adc = [
    epg.Imaging(pixels, voxel_size=pixsize, phase=-rf[i].phi, reduce=(1, 2))
    for i in range(nphase)
]
# T1/T2 and time accumulation (for T2* and B0)
rlx1 = epg.E([i * TR / nread for i in range(nread)], [T1], [T2])
rlx1 *= epg.C(rlx1.tau)  # time accumulation
rlx2 = epg.E([TR * (nread - 1 - i) / nread for i in range(nread)], [T1], [T2])
rlx2 *= epg.C(rlx2.tau)  # time accumulation
# readout gradient
kx = np.array([2 * np.pi / FOV, 0])  # rad/m
gx1 = epg.S([kx * (i - nread / 2) for i in range(nread)])
gx2 = epg.S([kx * (nread - i + 1) for i in range(nread)])
gxspl = epg.S(1.5 * kx * nread / 2)
# phase encoding gradients
kp = np.array([0, 2 * np.pi / FOV])  # rad/m
gp1 = [epg.S(kp * i) if i != 0 else epg.NULL for i in range(-nphase // 2, nphase // 2)]
gp2 = [epg.S(-kp * i) if i != 0 else epg.NULL for i in range(-nphase // 2, nphase // 2)]
# build sequence
seq = [init] + [
    [rf[i], gx1, gp1[i], rlx1, adc[i], rlx2, gx2, gxspl, gp2[i]] for i in range(nphase)
]
# simulate
sig_epg, time_epg = {}, {}
for tol in [1e-1, 1e-2, 1e-8]:
    print(f"EPG with tol={tol}")
    tic = time.time()
    # also return number of phase states
    kspace, nstates = epg.simulate(
        # seq, max_nstate=5, kgrid=1e-8, disp=True, probe=(None, "nstate"),
        seq,
        prune=tol,
        kgrid=1e-8,
        disp=True,
        probe=(None, "nstate"),
    )
    duration = time.time() - tic
    # FFT
    sig = np.fft.fftshift(np.fft.fft2(np.asarray(kspace))) / nread
    # store
    nstate = np.max(nstates)
    sig_epg[nstate] = sig
    time_epg[nstate] = duration
    print(f"duration={duration:.1f}")

# isochromats
sig_iso, time_iso = {}, {}
for niso in [10, 100, 1000]:
    print(f"Isochromats with n={niso}")
    # isochromats positions
    iso = random.uniform(-0.5, 0.5, (niso, 2)) * pixsize
    # isochromats off-resonance frequencies (num. cycle), Cauchy distribution
    # omega = np.tan(0.999 * np.pi * random.uniform(-0.5, 0.5, niso)) / 2 / np.pi
    omega = np.tan(0.999 * np.pi * np.linspace(-0.5, 0.5, niso)) / 2 / np.pi
    # set proton density
    init = epg.PD(pds)
    # rf pulse and ADC with phase offset
    rf = [epg.T(FA, 0)] * nphase
    # rf = [epg.T(FA, 117 * i * (i + 1)/2) for i in range(nphase)] # rf spoiling
    adc = [epg.Adc(reduce=True, phase=-rf[i].phi) for i in range(nphase)]
    # T1/T2/T2 (3 x 1 x niso)
    rlx = epg.E(TR / nread, T1, T2)
    rlx *= epg.P(TR / nread, 1 / np.reshape(T2p, (3, 1, 1)) * omega)  # T2'
    # gradients (2 x 1 x npix x niso)
    g = (pixels + iso[:, NAX]).T[:, NAX] / FOV  # (num cycles)
    # readout gradient
    gxpre = epg.P(1, -g[0] * nread / 2)
    gx = epg.P(1, g[0])
    gxspl = epg.P(1, 1.5 * g[0] * nread / 2)
    # phase encoding gradients
    gp1 = [
        epg.P(1, g[1] * i) if i != 0 else epg.NULL
        for i in range(-nphase // 2, nphase // 2)
    ]
    gp2 = [
        epg.P(1, -g[1] * i) if i != 0 else epg.NULL
        for i in range(-nphase // 2, nphase // 2)
    ]
    # build sequence
    seq = [init] + [
        [rf[i], gxpre, gp1[i]] + [adc[i], rlx, gx] * nread + [gxspl, gp2[i]]
        for i in range(nphase)
    ]
    # simulate
    tic = time.time()
    kspace = epg.simulate(seq, disp=True) / niso
    duration = time.time() - tic
    # FFT
    sig = np.fft.fftshift(np.fft.fft2(kspace.reshape(nphase, nread))) / nread
    # store
    sig_iso[niso] = sig
    time_iso[niso] = duration
    print(f"duration={duration:.1f}")

#
# plot
max_epg = sig_epg[max(sig_epg)]
vmin, vmax = [func(np.abs(max_epg)) for func in (np.min, np.max)]
fig, axes = plt.subplots(nrows=2, ncols=3, num="iso-vs-epg-2d", figsize=(8, 7))
for i, nstate in enumerate(sig_epg):
    plt.sca(axes[0, i])
    plt.imshow(np.abs(sig_epg[nstate]), cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"EPG\n(n.states: {nstate}, dur.: {time_epg[nstate]/60:.1f}min)")
    plt.axis("off")
for i, niso in enumerate(sig_iso):
    plt.sca(axes[1, i])
    plt.imshow(np.abs(sig_iso[niso]), cmap="gray", vmin=vmin, vmax=vmax)
    plt.title(f"Isochromats\n(n.iso: {niso}, dur.: {time_iso[niso]/60:.1f}min)")
    plt.axis("off")
plt.suptitle(f"Isochromats vs EPG")
plt.tight_layout()
# plt.show()
plt.savefig(fig.get_label() + ".png", dpi=200)
