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
from epgpy import epg, stats, diff


# define MRF sequence
nTR = 400
T1, T2 = 1380, 80

# operators
adc = epg.ADC
grd = epg.S(1)
inv = epg.T(180, 0) # inversion
rlx0 = epg.E(20, T1, T2, order1=['T1', 'T2'], duration=True) # inversion delay

# parameters to optimize
alphas = [f'alpha_{i:03d}' for i in range(nTR)]
taus = [f'tau_{i:03d}' for i in range(nTR)]

# cross derivatives to compute in RF operator
order2_rf = [[('T1', alphas[i]), ('T2', alphas[i])] for i in range(nTR)]

def rf(i, alpha, deriv=False):
    if not deriv:
        return epg.T(alpha, (i%2) * 180)
    return epg.T(
        alpha, (i%2) * 180, 
        order1={alphas[i]: 'alpha'}, # use parameter alias
        order2=order2_rf[i], # select cross derivatives
        )
    
# cross derivatives to compute in relaxation operator
order2_rlx = [[('T1', taus[i]), ('T2', taus[i])] for i in range(nTR)]
def rlx(i, tau, deriv=False): 
    if not deriv:
        return epg.E(tau, T1, T2, order1=['T1', 'T2'], duration=True)
    return epg.E(
        tau, T1, T2, 
        order1={'T1': 'T1', 'T2': 'T2', taus[i]: 'tau'}, # use parameter aliases
        order2=sum(order2_rlx + order2_rf, start=[]), # select cross derivatives
        duration=True,
    )

# sequence generator function
def sequence(angles, times, deriv=False):
    return [inv, rlx0] + [
        [rf(i, angles[i], deriv), rlx(i, times[i], deriv), adc, grd] 
        for i in range(nTR)
    ]



""" optimize sequence parameters

# constraints
alpha in [10, 60]
TR in [11, 16]
|alpha_i - alpha_i+1| < 1

# W = diag(1, 1/T1^2, 1/T2^2)

"""
# maximum number of phase states
nstate = 10
# CLRB weights
weights = [1, 1 / T1 ** 2, 1 / T2 ** 2]
# Probe operators for Jacobien and Hessian
Jac = epg.Jacobian(['magnitude', 'T1', 'T2'])
Hes = epg.Hessian(['magnitude', 'T1', 'T2'], alphas + taus)

def Num(sm):
    # return number of computed partials derivatives
    return (len(sm.order1), len(sm.order2))

# def condition(sm, th=1e-5):
#     return sm.arrays.apply('states', lambda states: np.abs(states).max() < th)
# remove phase states with little signal
pruner = diff.PartialsPruner(condition=1e-6, variables=alphas + taus)

# store parameters at each iteration
iterations = []

def callback(params):
    """ store parameters """
    global iterations
    iterations.append(params)

def signal(params):
    alphas, taus = params[:nTR], params[nTR:]
    return epg.simulate(sequence(alphas, taus), max_nstate=nstate)


def costfun(params):
    """crlb cost function"""
    alphas, taus = params[:nTR], params[nTR:]
    jac = epg.simulate(sequence(alphas, taus), probe=Jac, max_nstate=nstate)
    cost = stats.crlb(np.moveaxis(jac, -1, 0), W=weights, log=True)
    return cost[0]


def costjac(params):
    """jacobian of cost function w/r to parameters alpha_xx and tau_xx"""
    alphas, taus = params[:nTR], params[nTR:]
    # simulate sequence
    jac, hes, num = epg.simulate(
        sequence(alphas, taus, deriv=True), 
        probe=[Jac, Hes, Num], 
        # callback=pruner, 
        max_nstate=nstate,
    )
    # CRLB
    cost, grad = stats.crlb(
        np.moveaxis(jac, -1, 0), 
        np.moveaxis(hes, -1, 0),
        W=weights,
        log=True,
    )
    niter = len(iterations)
    elaps = time.time() - tic
    print(
        f"({niter}) crlb={cost[0]:.8f} "
        f"(elapsed time: {elaps:.0f}s, "
        f"num. partials order 1/2: {max(num[:, 0])}/{max(num[:, 1])})",
    )
    return grad.ravel()


def constraint_function(params):
    """constraint function: |alpha_i - alpha_i+1| < 1"""
    return (1 - np.max(np.abs(np.diff(params[:nTR])))) * 10


# initial FA between 10 and 60
rstate = np.random.RandomState(0)
nFA = 300
init_FA = []
for i in range(nTR//nFA + 1):
    FA = np.sin(np.arange(1, 1 + nFA) * np.pi / nFA) * (60 - 10) + 10
    FA[-10:] = 0
    init_FA.extend(FA.tolist())
init_FA = init_FA[:nTR]

# initial TR: Perlin noise 11.5 to 14.5 
def smoothstep(x):
    s = x - np.floor(x)
    return 3* s**2 - 2 * s**3
Np = nTR // 10 # 10 random values
coords = np.arange(nTR)
rdm = rstate.uniform(-1, 1, (nTR - 1) // Np + 2)
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
    "options": {"disp": True, "ftol": 1e-7, "maxiter": 250},
}

# optimize
print(f'Optimize MRF sequence')
print(f'Num. TR: {nTR}')
print(f'Num. states: {nstate}')
print(f'Num. parameters (alpha/tau): {len(init)}')
print(f'Initial cost: {costfun(init):.8f}')

tic = time.time()
res = optimize.minimize(costfun, init, **config)

duration = time.time() - tic
print(f"Results: alphas={res.x[:nTR]}, taus={res.x[nTR:]}")
print(f"Optimization time: {duration/60:.1f} min")


#
# plot

fig, axes = plt.subplots(nrows=2, num="mrf-optim", sharex=True)
initconst = constraint_function(init)
resconst = constraint_function(res.x)

plt.sca(axes.flat[0])
plt.plot(init[:nTR], label=f"initial (constraint cost: {initconst:.2f})")
plt.plot(res.x[:nTR], label=f"optimized (constraint cost: {resconst:.2f})")
plt.legend()
plt.title("Flip angle values")
plt.ylabel("flip angle (degree)")
plt.ylim(0, 70)
plt.sca(axes.flat[1])
plt.plot(init[nTR:], label="initial")
plt.plot(res.x[nTR:], label="optimized")
plt.legend()
plt.title("TE values")
plt.ylabel("TE (ms)")
plt.xlabel("echo index")
plt.ylim(10, 17)
plt.tight_layout()


plt.figure("mrf-fingerprint")
sig0 = signal(init)
crb0 = costfun(init)
sig1 = signal(res.x)
crb1 = costfun(res.x)
plt.plot(np.abs(sig0[:, 0]), label=f"initial (log10(CRB): {crb0[0]:0.2f})")
plt.plot(np.abs(sig1[:, 0]), label=f"optimized (log10(CRB): {crb1[0]:0.2f})")
plt.xlabel("echo index")
plt.ylabel("signal")
plt.title(f"Fingerprint for T1={T1}ms, T2={T2}ms")
plt.legend()
plt.grid()
plt.tight_layout()


plt.figure("mfr-iterations")
jacs = [
    epg.simulate(sequence(params[:nTR], params[nTR:]), probe=Jac, max_nstate=10)[..., 0]
    for params in iterations
]
crb_tot = stats.crlb(jacs, W=weights, log=False)
crb_mag, crb_T1, crb_T2 = stats.crlb_split(jacs, W=weights, log=False)
plt.plot(crb_mag, ":", label="CRB magnitude")
plt.plot(crb_T1, ":", label="CRB T1")
plt.plot(crb_T2, ":", label="CRB T2")
plt.plot(crb_tot, label="CRB total")
plt.title("Evolution of CRB during optimization")
plt.xlabel("Iteration index")
plt.ylabel("log10(CRB)")
plt.legend()
plt.tight_layout()

plt.show()
