"""Run Hessian calculation for an MRF sequence (cf. examples/differentiation/optim_mrf.py)

Same as `differentiation_mrf.py`, but using the `epg.sequence` module
"""

import time
import numpy as np
from epgpy.sequence import Sequence, operators, repeat

# define MRF sequence
nTR = 400
T1, T2 = 1380, 80

# define parameters to optimize
alphas = [f"alpha_{i:03d}" for i in range(nTR)]
TRs = [f"TR_{i:03d}" for i in range(nTR)]

# operators
adc = operators.ADC
spl = operators.S(1)
rf = operators.T("alpha", 90)
rlx = operators.E("TR", "T1", "T2")

# MRF sequence
seq = Sequence(repeat([rf, rlx, adc, spl], alpha=alphas, TR=TRs))

# random flip angle and TR values
values_alphas = dict(zip(alphas, np.random.uniform(10, 60, nTR)))
values_TRs = dict(zip(TRs, np.random.uniform(11, 16, nTR)))

# build hessian function
hessfunc = seq.hessian(
    ["magnitude", "T1", "T2"],  # hessian "rows"
    alphas + TRs,  # hessian "columns"
    options={"max_nstate": 10, "disp": True},
)

print(f"Simulate MRF sequence (nTR={nTR})")
tic = time.time()
sig, grad, hess = hessfunc({**values_alphas, **values_TRs}, T1=T1, T2=T2)

toc = time.time()
print(f"Done. Duration: {toc - tic:.1f}s")
