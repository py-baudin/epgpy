""" Multi-spin echo simulation and profiling

Simulate several T2 values and B1-attenuations at once through vectorization.

"""

import time
import numpy as np
from epgpy import operators, functions

# sequence
necho = 18
TE = 9.5

num_t2 = 100
num_b1 = 100

# parameters
T1 = 1400
T2 = np.linspace(20, 60, num_t2)
att = np.linspace(0.2, 1, num_b1)

# build sequence
exc = operators.T(90, 90)
shift = operators.S(1)
rfc = operators.T(180 * att, 0)  # B1 on 1st axis
rlx = operators.E(TE / 2, T1, [T2])  # put T2 on 2d axis
adc = operators.ADC
seq = [exc] + [shift, rlx, rfc, shift, rlx, adc] * necho

shape = functions.getshape(seq)
nops = len(seq)
print(f"Simulate: {nops} operators, {necho} time points, {np.prod(shape)} signals")

time0 = time.time()

# simulate
sim = functions.simulate(seq)

duration = time.time() - time0

print("Done.")
print(f"Duration: {duration: 0.2f}s")
