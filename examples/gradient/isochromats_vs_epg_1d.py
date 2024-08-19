import numpy as np
import epgpy as epg
from matplotlib import pyplot as plt

NAX = np.newaxis
random = np.random #.RandomState(0)


# acquisition
FA = 60 # degrees
TR = 10 # ms
FOV = 200e-3 # m
nread = 64 
pixsize = FOV / nread
# pixel positions
pixels = np.arange(-nread//2, nread//2) / nread * FOV

# system
# proton density
pd = random.uniform(size=nread)
# relaxation
T1, T2 = 830, 70 # ms
T2p = 30 # ms (T2 prime)

# EPG
print('EPG')
adc = epg.Imaging(pixels, voxel_size=pixsize) 
init = epg.System(weights=pd, modulation=-1/T2p)
rf = epg.T(FA, 90)
rlx = epg.E(TR, T1, T1)
rlx *= epg.C(TR) # time accumulation
# readout gradient 
k = 2 * np.pi / FOV # rad/m
gxpre = epg.S(-k * nread/2)
gx = epg.S(k) 
seq = [init, rf, gxpre] + [adc, rlx, gx] * nread
# simulate
kspace = epg.simulate(seq, kgrid=0.1)
sig_epg = np.fft.fftshift(np.fft.fft(kspace)) / nread

# isochromats
sig_iso = {}
for niso in [10, 100, 1000, 10000]:
    print(f'Isochromats with n={niso}')
    # isochromats positions
    iso = random.uniform(-0.5, 0.5, niso) * pixsize
    # isochromats off-resonance frequencies (num. cycle)
    omega = np.tan(0.999 * np.pi * np.linspace(-0.5, 0.5, niso)) / 2 / np.pi
    # omega = np.tan(0.999 * np.pi * random.uniform(-0.5, 0.5, niso)) / 2 / np.pi
    adc = epg.ADC
    init = epg.PD(pd)
    rf = epg.T(FA, 90)
    rlx = epg.E(TR, T1, T1)
    rlx *= epg.P(TR, 1/T2p * omega[NAX]) # T2' 
    # readout gradient 
    g = (pixels[:, NAX] + iso) / FOV # (num cycles)
    gxpre = epg.P(1, -g * nread / 2 ) 
    gx = epg.P(1, g) 
    # simulate
    seq = [init, rf, gxpre] + [adc, rlx, gx] * nread
    sim_iso = epg.simulate(seq)
    kspace = np.sum(sim_iso.sum(axis=-1), axis=-1) / niso
    sig = np.fft.fftshift(np.fft.fft(kspace)) / nread
    sig_iso[niso] = sig

#
# plot
fig, axes = plt.subplots(ncols=2, sharex=True, num='iso-vs-epg-1d', figsize=(8, 5))
colors = plt.cm.viridis(np.linspace(0, 1, len(sig_iso)))
plt.sca(axes.flat[0])
for i, niso in enumerate(sig_iso):
    plt.plot(1e3*pixels, np.abs(sig_iso[niso]), label=f'Bloch (num. iso.: {niso})', color=colors[i], alpha=0.5)
plt.plot(1e3*pixels, np.abs(sig_epg), 'r:+', label='EPG')
plt.legend()
plt.xlabel('location (mm)')
plt.ylabel('Magnitude (a.u.)')
plt.sca(axes.flat[1])
for i, niso in enumerate(sig_iso):
    plt.plot(1e3*pixels, np.angle(sig_iso[niso]), label=f'Bloch (num. iso.: {niso})', color=colors[i], alpha=0.5)
plt.plot(1e3*pixels, np.angle(sig_epg), 'r:+', label='EPG')
plt.xlabel('location (mm)')
plt.ylabel('Phase (rad.)')
plt.suptitle(f'Isochromats vs EPG\n(T1={T1}ms, T2={T2}ms, T2\'={T2p}ms)')
plt.tight_layout()
plt.show()

