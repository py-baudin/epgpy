import numpy as np
from epgpy.sequence import Sequence, Variable, operators
from epgpy import operators as epgops, functions, stats

print("define sequence")

# define MRF sequence
nTR = 400
TE = 3 # ms
T1, T2 = 1380, 80

# operators
adc = operators.ADC
spl = operators.S(1)
inv = operators.T(180, 90) # inversion
rlx0 = operators.E(20, 'T1', 'T2', duration=True) # initial relaxation
rlx1 = operators.E(TE, 'T1', 'T2', duration=True) # before adc
rf = lambda alpha: operators.T(alpha, 90)
rlx2 = lambda TR: operators.E(Variable(TR) - TE, 'T1', 'T2', duration=True)

# parameters to optimize
alphas = [f'alpha_{i:03d}' for i in range(nTR)]
TRs = [f'TR_{i:03d}' for i in range(nTR)]

# build sequence
seq = Sequence(
    [[inv, rlx0], [[rf(alphas[i]), rlx1, adc, rlx2(TRs[i]), spl] for i in range(nTR)]],
    options={'max_nstate': 10},
)

# build and simulate
TR_values = list(np.random.uniform(4, 10, nTR))
alpha_values = list(np.random.uniform(5, 30, nTR))
values = dict(zip(alphas + TRs, TR_values + alpha_values))

print('build')
order1 = ['magnitude', 'T1', 'T2'] + alphas + TRs
order2 = [(v1, v2) for v2 in alphas + TRs for v1 in ['magnitude', 'T1', 'T2']]
ops = seq.build({'T1': T1, 'T2': T2, **values}, order1=order1, order2=order2)

probes = [
    epgops.Jacobian(['magnitude', 'T1', 'T2']), 
    epgops.Hessian(['magnitude', 'T1', 'T2'], alphas + TRs)
]
print('simulate')
jac, hess = functions.simulate(ops, probe=probes, max_nstate=10)
jac = np.moveaxis(jac, -1, 0)
hess = np.moveaxis(hess, -1, 0)

# crlb
weights = np.asarray([1, 1/T1**2, 1/T2**2])
print('crlb')
crlb1, dcrlb1 = stats.crlb(jac, hess, W=weights)

print('build, sim, crlb')
# _, jac2, hess2 = seq.hessian(['magnitude', 'T1', 'T2'], alphas + TRs)(values, T1=T1, T2=T2)
crlb2, dcrlb2 = seq.crlb(['magnitude', 'T1', 'T2'], gradient=alphas + TRs, weights=weights)(values, T1=T1, T2=T2)

# assert np.allclose(crlb1, crlb2)
print('done')