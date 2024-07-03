import time
import numpy as np
from epgpy import epg


# define MRF sequence
nTR = 100
T1, T2 = 1380, 80

alphas = [f'alpha_{i:03d}' for i in range(nTR)]
taus = [f'tau_{i:03d}' for i in range(nTR)]
order2_rf = [[('T1', alphas[i]), ('T2', alphas[i])] for i in range(nTR)]
order2_rlx = [[('T1', taus[i]), ('T2', taus[i])] for i in range(nTR)]

def rf(i, alpha):
    return epg.T(
        alpha, 90, 
        order1={alphas[i]: {'alpha': 1}}, 
        order2=order2_rf[i],
        )
    
def rlx(i, tau): 
    return epg.E(
        tau, T1, T2, 
        order1={'T1': {'T1': 1}, 'T2': {'T2': 1}, taus[i]: {'tau': 1}},
        order2=sum(order2_rlx + order2_rf, start=[]),
        duration=True,
    )

grd = epg.S(1)
adc = epg.ADC
def sequence(angles, times):
    """ MRF sequence"""
    return [
        [rf(i, angles[i]), rlx(i, times[i]), adc, grd] 
        for i in range(nTR)
    ]

# simulate with order2 derivatives
angles = np.random.uniform(10, 60, nTR)
times = np.random.uniform(11, 16, nTR)

Jac = epg.Jacobian(['magnitude', 'T1', 'T2'])
Hes = epg.Hessian(['magnitude', 'T1', 'T2'], alphas + taus)
def Num(sm):
    return (len(sm.order1), len(sm.order2))

print(f'Simulate MRF sequence (nTR={nTR})')
tic = time.time()
jac, hes, num = epg.simulate(sequence(angles, times), probe=[Jac, Hes, Num])

toc = time.time()
print(f'Done. Duration: {toc - tic:.1f}s, num. partials order1: {max(num[:, 0])}, num. partials order2: {max(num[:, 1])}.')
