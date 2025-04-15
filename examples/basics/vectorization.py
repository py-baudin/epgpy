""" Vectorization and reduction examples """

import numpy as np
import epgpy as epg

# MSE sequence
necho = 20
exc = epg.T(90, 90)
spl = epg.S(1)
adc = epg.ADC
# simulate 5 off-resonance frequencies
rlx = epg.E(5, 1400, 30, [-0.2, -0.1, 0, 0.1, 0.2])
# simulate 3 flip angles
inv = epg.T([[120, 150, 180]], 0)

# vectorization

# the output reflects the parameter's shape
seq = [exc] + [[rlx, spl, inv, spl, rlx, adc]] * 20
assert epg.getshape(seq) == (5, 3)

signal = epg.simulate(seq)
assert signal.shape == (necho, 5, 3)


# reduce with Adc

# signal is reduced (summed) on selected axis/axes
signal_reduced = epg.simulate(seq, probe=epg.Adc(reduce=0))
assert signal_reduced.shape == (20, 3)
assert np.allclose(signal_reduced, signal.sum(axis=1))

signal_reduced = epg.simulate(seq, probe=epg.Adc(reduce=(0, 1)))
assert signal_reduced.shape == (20,)
assert np.allclose(signal_reduced, signal.sum(axis=(1, 2)))

# set weights (w/ automatic reduce)
weights = [i / 10 for i in range(5)]
signal_weights = epg.simulate(seq, probe=epg.Adc(weights=weights))
assert signal_weights.shape == (20, 3)
assert np.allclose(signal_weights, np.vecdot(weights, signal, axes=[0, 1]))

# set weigths with operator PD
pd = epg.PD(weights)
signal_pd = epg.simulate([pd] + seq, probe=epg.Adc(reduce=0))
assert np.allclose(signal_pd, signal_weights)
