"""Multi-spin echo simulation and profiling

Simulate several T2 values and B1-attenuations at once through vectorization.

"""
import time
import numpy as np
import epgpy as epg

num_t2 = 200  # number of T2 values
num_b1 = 200  # number of B1 attenuation values

# parameters
necho = 18
TE = 9.5
T1 = 1400
T2 = np.linspace(20, 60, num_t2)
att = np.linspace(0.2, 1, num_b1)

# build sequence
exc = epg.T(90, 90)
shift = epg.S(1)
rfc = epg.T(180 * att, 0)  # B1 on 1st axis
rlx = epg.E(TE / 2, T1, [T2])  # put T2 on 2d axis
adc = epg.ADC
seq = [exc] + [shift, rlx, rfc, shift, rlx, adc] * necho

# simulate
shape = epg.getshape(seq)  # get signal's shape
print(f"Simulate {np.prod(shape)} signals")

time0 = time.time()
signal = epg.simulate(seq, disp=True)
duration = time.time() - time0

print(f"Duration: {duration:.2f}s")
print(f"Output shape: {signal.shape}")

if epg.get_array_module().__name__ == 'numpy':
    print('(You may consider using `cupy` with `export ARRAY_MODULE=cupy`)')
