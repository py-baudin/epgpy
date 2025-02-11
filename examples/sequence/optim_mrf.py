"""
# optimization of an MRF sequence

Example taken from:

> Lee PK, Watkins LE, Anderson TI, Buonincontri G, Hargreaves BA:
  Flexible and Efficient Optimization of Quantitative Sequences using Automatic
  Differentiation of Bloch Simulations.
  Magn Reson Med 2019; 82:1438–1451.

"""

import time
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from epgpy.sequence import Sequence, Variable, operators
from epgpy import stats

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



""" optimize sequence parameters

# constraints
alpha in [10, 60]
TR in [11, 16]
|alpha_i - alpha_i+1| < 1

# W = diag(1, 1/T1^2, 1/T2^2)

"""

    
# CLRB options
options = {'sigma2': 1e1, 'log': False}
weights = [1, 1 / T1 ** 2, 1 / T2 ** 2]
targets = ['magnitude', 'T1', 'T2']

# store parameters at each iteration
iterations = []

def callback(params):
    """ store parameters """
    global iterations
    iterations.append(params)

def signal(params):
    values = dict(zip(alphas + TRs, params))
    return seq.signal(values, T1=T1, T2=T2)


def costfun(params):
    """crlb cost function"""
    values = dict(zip(alphas + TRs, params))
    cost = seq.crlb(targets, weights=weights, **options)(values, T1=T1, T2=T2)
    return cost[0]


def costjac(params):
    """jacobian of cost function w/r to parameters alpha_xx and tau_xx"""
    values = dict(zip(alphas + TRs, params))
    cost, grad = seq.crlb(targets, gradient=alphas + TRs, weights=weights, **options)(values, T1=T1, T2=T2)

    # display
    niter = len(iterations)
    elaps = time.time() - tic
    print(
        f"({niter}) crlb={cost[0]:.8f} "
        f"(elapsed time: {elaps:.0f}s)"
    )
    return grad.ravel()


def constraint_function(params):
    """constraint function: |alpha_i - alpha_i+1| < 1"""
    diff = np.diff(params, prepend=params[0])
    diff[nTR:] = 0
    return 1 - np.abs(diff)

# initial FA between 10 and 60
random = np.random.RandomState(0)
nFA = 300
init_FA = []
for i in range(nTR//nFA + 1):
    FA = np.sin(np.arange(1, 1 + nFA) * np.pi / nFA) * (60 - 10) + 10
    FA[-10:] = 0
    init_FA.extend(FA.tolist())
init_FA = np.clip(init_FA[:nTR], 10, 60)


# initial TR: Perlin noise 11.5 to 14.5 
def smoothstep(x):
    s = x - np.floor(x)
    return 3* s**2 - 2 * s**3
Np = nTR // 10 # 10 random values
coords = np.arange(nTR)
rdm = random.uniform(-1, 1, (nTR - 1) // Np + 2)
coeff1 = rdm[coords // Np] * (coords - Np * (coords // Np))
coeff2 = rdm[coords // Np + 1] * (coords - Np * (coords // Np + 1))
init_TR = coeff1 + smoothstep(coords/Np)*(coeff2 - coeff1)
init_TR = init_TR / (Np / 2) * (14.5 - 11.5) / 2 + (14.5 + 11.5) / 2

# initial parameters
init = list(init_FA) + list(init_TR)

# boundaries
bounds = [(10, 60)] * nTR + [(11, 16)] * nTR

config = {
    "method": "SLSQP",
    "jac": costjac,
    "bounds": bounds,
    "constraints": [{"type": "ineq", "fun": constraint_function}],
    "callback": callback,
    "options": {"disp": True, "ftol": 1e-6, "maxiter": 250},
}

# optimize
print(f'Optimize MRF sequence')
print(f'Num. TR: {nTR}')
print(f'Num. states: {seq.options['max_nstate']}')
print(f'Num. parameters (alpha/tau): {len(init)}')
print(f'Initial cost: {costfun(init):.8f}')

tic = time.time()
res = optimize.minimize(costfun, init, **config)

duration = time.time() - tic
print(f"Optimization time: {duration/60:.1f} min")

# compute crlb for each parameters
jacs = [
    seq.jacobian(['T1', 'T2'])(dict(zip(alphas + TRs, params)), T1=T1, T2=T2)[0]
    for params in iterations
]
crb_tot = stats.crlb(jacs, W=weights, **options)
crb_mag, crb_T1, crb_T2 = stats.crlb_split(jacs, W=weights, **options)

#
# plot
plt.close('all')

fig, axes = plt.subplots(nrows=2, num="mrf-optim", sharex=True)
plt.sca(axes.flat[0])
plt.plot(init[:nTR], label=f"initial")
plt.plot(res.x[:nTR], label=f"optimized")
plt.legend(loc='upper right')
plt.ylabel("flip angle [degree]")
plt.xlim(0, nTR)
plt.ylim(0, 70)
plt.sca(axes.flat[1])
plt.plot(init[nTR:], label="initial")
plt.plot(res.x[nTR:], label="optimized")
plt.ylabel("TR [ms]")
plt.xlabel("echo index")
plt.ylim(10, 17)
plt.suptitle('Sequence parameters')
plt.tight_layout()


plt.figure("mrf-fingerprint")
sig0 = signal(init)
crb0 = costfun(init)
sig1 = signal(res.x)
crb1 = costfun(res.x)
plt.plot(np.abs(sig0[:, 0]), label=f"initial")
plt.plot(np.abs(sig1[:, 0]), label=f"optimized")
plt.xlabel("echo index")
plt.ylabel("signal [a.u.]")
plt.title(f"MR ingerprint for T1={T1}ms, T2={T2}ms")
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()


plt.figure("mrf-iterations")
plt.plot(crb_mag, ":", label="CRLB M0")
plt.plot(crb_T1, ":", label="CRLB T1")
plt.plot(crb_T2, ":", label="CRLB T2")
plt.plot(crb_tot, label="CRLB total")
plt.title(f"CRLB optimization (duration: {duration/60:.0f} min)")
plt.xlabel("Iteration index")
plt.ylabel("CRLB")
plt.legend(loc='upper right')
plt.tight_layout()

plt.show()

# for fig in plt.get_fignums():
#     fig = plt.figure(fig)
#     filename = fig.get_label().replace('-', '_').replace(' ', '_') + '.png'
#     plt.savefig(filename, dpi=200)
