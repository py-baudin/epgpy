"""Run Hessian calculation for an MRF sequence (cf. examples/differentiation/optim_mrf.py)"""

import time
import numpy as np
import epgpy as epg
from epgpy import diff


# define MRF sequence
nTR = 400
T1, T2 = 1380, 80

# define variables
alphas = [f"alpha_{i:03d}" for i in range(nTR)]
TRs = [f"tau_{i:03d}" for i in range(nTR)]

# select cross derivatives to compute
order2_rf = [[("T1", alphas[i]), ("T2", alphas[i])] for i in range(nTR)]
order2_rlx = [[("T1", TRs[i]), ("T2", TRs[i])] for i in range(nTR)]

# operators
adc = epg.ADC
spl = epg.S(1)


def rf(i, alpha):
    return epg.T(
        alpha,
        90,
        order1={alphas[i]: "alpha"},  # use parameter alias
        order2=order2_rf[i],  # select cross derivatives
    )


def rlx(i, tau):
    return epg.E(
        tau,
        T1,
        T2,
        order1={"T1": "T1", "T2": "T2", TRs[i]: "tau"},  # use parameter aliases
        order2=sum(order2_rlx + order2_rf, start=[]),  # select cross derivatives
        duration=True,
    )


# MRF sequence
def sequence(angles, times):
    return [[rf(i, angles[i]), rlx(i, times[i]), adc, spl] for i in range(nTR)]


# random flip angle and TR values
values_alphas = np.random.uniform(10, 60, nTR)
values_TRs = np.random.uniform(11, 16, nTR)

# Select Hessian variables ("rows" and "columns")
Hes = epg.Hessian(["magnitude", "T1", "T2"], alphas + TRs)

print(f"Simulate MRF sequence (nTR={nTR})")
tic = time.time()
hes = epg.simulate(
    sequence(values_alphas, values_TRs),
    probe=Hes,
    max_nstate=10,
    disp=True,
)

toc = time.time()
print(f"Done. Duration: {toc - tic:.1f}s")
