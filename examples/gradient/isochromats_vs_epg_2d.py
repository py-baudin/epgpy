import pathlib
import time
import numpy as np
import epgpy as epg
from matplotlib import pyplot as plt

NAX = np.newaxis
random = np.random #.RandomState(0)

# load brain phantom
HERE = pathlib.Path(__file__).parent
wm, gm, csf = np.load(HERE / 'brain.npy')

# tmp
from scipy import ndimage
zoom = 64 / wm.shape[0]
wm = ndimage.zoom(wm, zoom)
gm = ndimage.zoom(gm, zoom)
csf = ndimage.zoom(csf, zoom)

mask = np.max([wm, gm, csf], axis=0) > 1e-5

# acquisition
FA = 60 # degrees
TR = 10 # ms
FOV = 200e-3 # m
nread, nphase = mask.shape

pixsize = FOV / nread
pixels = np.mgrid[-nread//2:nread//2, -nphase//2:nphase//2] 
pixels = pixels.reshape(2, -1).T * FOV / [nread, nphase]

# system
# GM, WM, CSF
PD = [0.8, 0.7, 1.0] # a.u.
T1 = [1.55e3, 0.83e3, 4.16e3] # ms
T2 = [0.09e3, 0.07e3, 1.65e3] # ms
T2p = [0.322e3, 0.183e3, 0.0591e3] # ms
pds = np.stack([gm * PD[0], wm * PD[1], csf * PD[2]]).reshape(3, -1)

# EPG
print('EPG')
adc = epg.Imaging(pixels, voxel_size=pixsize) 
init = epg.System(weights=pds, modulation=-1/np.array(T2p))
rf = epg.T(FA, 90)
rlx = epg.E(TR, T1, T1)
rlx *= epg.C(TR) # time accumulation
# readout gradient 
kx = np.array([2 * np.pi / FOV, 0]) # rad/m
kp = np.array([0, 2 * np.pi / FOV]) # rad/m
gxpre = epg.S(-kx * nread/2)
gx = epg.S(kx)
gxspl = epg.S(1.5 * kx * nread/2)
gp1 = [epg.S(kp * i) if i !=0 else epg.NULL for i in range(-nphase//2, nphase//2)]
gp2 = [epg.S(-kp * i) if i !=0 else epg.NULL for i in range(-nphase//2, nphase//2)]
seq = [init] + [[rf, gxpre, gp1[i]] + [adc, rlx, gx] * nread + [gxspl, gp2[i]] for i in range(nphase)]
# simulate
sig_epg, time_epg = {}, {}
for tol in [5e-2, 1e-2, 1e-8]:
    print(f'EPG with tol={tol}')
    tic = time.time()
    kspace, nstates = epg.simulate(seq, prune=tol, kgrid=1, disp=True, probe=(adc, 'nstate'))
    duration = time.time() - tic
    sig = np.fft.fftshift(np.fft.fft2(kspace.reshape(nphase, nread))) / nread
    nstate = max(nstates)
    sig_epg[nstate] = sig
    time_epg[nstate] = duration


# isochromats
sig_iso, time_iso = {}, {}
for niso in [10, 100, 101]:#, 10000]:
    print(f'Isochromats with n={niso}')
    # isochromats positions
    iso = random.uniform(-0.5, 0.5, niso) * pixsize
    # isochromats off-resonance frequencies (num. cycle)
    omega = np.tan(0.999 * np.pi * np.linspace(-0.5, 0.5, niso)) / 2 / np.pi
    # omega = np.tan(0.999 * np.pi * random.uniform(-0.5, 0.5, niso)) / 2 / np.pi
    adc = epg.Adc(reduce=(0, 1, 2))
    init = epg.PD(pds)
    rf = epg.T(FA, 90)
    rlx = epg.E(TR, T1, T1)
    rlx *= epg.P(TR, 1/np.reshape(T2p,(3,1,1)) * omega) # T2' 
    # readout gradient 
    g = (pixels.T[:, NAX, ..., NAX] + iso) / FOV # (num cycles)
    gxpre = epg.P(1, -g[0] * nread / 2 ) 
    gx = epg.P(1, g[0]) 
    gxspl = epg.P(1, 1.5 * g[0] * nread / 2)
    gp1 = [epg.P(1, g[1] * i) if i !=0 else epg.NULL for i in range(-nphase//2, nphase//2)]
    gp2 = [epg.P(1, -g[1] * i) if i !=0 else epg.NULL for i in range(-nphase//2, nphase//2)]
    # simulate
    seq = [init] + [[rf, gxpre, gp1[i]] + [adc, rlx, gx] * nread + [gxspl, gp2[i]] for i in range(nphase)]
    tic = time.time()
    sim_iso = epg.simulate(seq, disp=True)
    duration = time.time() - tic
    kspace = sim_iso / niso
    sig = np.fft.fftshift(np.fft.fft2(kspace.reshape(nphase, nread))) / nread
    sig_iso[niso] = sig
    time_iso[niso] = duration

#
# plot
fig, axes = plt.subplots(nrows=2, ncols=3, num='iso-vs-epg-2d', figsize=(8,7))
for i, nstate in enumerate(sig_epg):
    plt.sca(axes[0, i])
    plt.imshow(np.abs(sig_epg[nstate]), cmap='gray')
    plt.title(f'EPG\n(n.states: {nstate}, dur.: {time_epg[nstate]/60:.1f}min)')
    plt.axis('off')
for i, niso in enumerate(sig_iso):
    plt.sca(axes[1, i])
    plt.imshow(np.abs(sig_iso[niso]), cmap='gray')
    plt.title(f'Isochromats\n(n.iso: {niso}, dur.: {time_iso[niso]/60:.1f}min)')
    plt.axis('off')
plt.suptitle(f'Isochromats vs EPG')
plt.tight_layout()
plt.show()

