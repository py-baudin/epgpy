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
TE = 3 # ms
T1, T2 = 1380, 80
phi = 90

# operators
adc = epg.ADC
grd = epg.S(1)
inv = epg.T(180, phi) # inversion
rlx0 = epg.E(20, T1, T2, order1=['T1', 'T2'], duration=True) # inversion delay


# parameters to optimize
alphas = [f'alpha_{i:03d}' for i in range(nTR)]
TRs = [f'TR_{i:03d}' for i in range(nTR)]

# cross derivatives to compute in RF operator
order2_rf1 = [('T1', alphas[i]) for i in range(nTR)]
order2_rf2 = [('T2', alphas[i]) for i in range(nTR)]

def rf(i, alpha, jac=False):
    if not jac:
        return epg.T(alpha, phi)
    return epg.T(
        alpha, phi, 
        order1={alphas[i]: 'alpha'}, # use parameter alias
        order2=[order2_rf1[i], order2_rf2[i]], # select cross derivatives
        )
    
# cross derivatives to compute in relaxation operator
order2_rlx = [('T1', TRs[i]) for i in range(nTR)] + [('T2', TRs[i]) for i in range(nTR)]
order2_rlx += order2_rf1 + order2_rf2

rlx1_nojac = epg.E(TE, T1, T2, order1=['T1', 'T2'], duration=True)
rlx1_jac = epg.E(TE, T1, T2, order1=['T1', 'T2'],  order2=order2_rlx, duration=True)
def rlx1(jac=False): 
    return rlx1_nojac if not jac else rlx1_jac

def rlx2(i, TR, jac=False): 
    if not jac:
        return epg.E(TR - TE, T1, T2, order1=['T1', 'T2'], duration=True)
    return epg.E(
        TR - TE, T1, T2, 
        order1={'T1': 'T1', 'T2': 'T2', TRs[i]: 'tau'}, # use parameter aliases
        order2=order2_rlx, # select cross derivatives
        duration=True,
    )

# sequence generator function
def sequence(angles, times, jac=False):
    return [inv, rlx0] + [
        [rf(i, angles[i], jac), rlx1(jac), adc, rlx2(i, times[i], jac), grd] 
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
Hes = epg.Hessian(['magnitude', 'T1', 'T2'], alphas + TRs)
sigma2 = 1e1

def Num(sm):
    # return number of computed partials derivatives
    return (len(sm.order1), len(sm.order2))

# def condition(sm, th=1e-5):
#     return sm.arrays.apply('states', lambda states: np.abs(states).max() < th)
# remove phase states with little signal
pruner = diff.PartialsPruner(condition=1e-6, variables=alphas + TRs)

# store parameters at each iteration
iterations = []

def callback(params):
    """ store parameters """
    global iterations
    iterations.append(params)

def signal(params):
    alphas, TRs = params[:nTR], params[nTR:]
    return epg.simulate(sequence(alphas, TRs), max_nstate=nstate)


def costfun(params):
    """crlb cost function"""
    alphas, TRs = params[:nTR], params[nTR:]
    jac = epg.simulate(sequence(alphas, TRs), probe=Jac, max_nstate=nstate)
    cost = stats.crlb(np.moveaxis(jac, -1, 0), W=weights, log=False, sigma2=sigma2)
    return cost[0]


def costjac(params):
    """jacobian of cost function w/r to parameters alpha_xx and tau_xx"""
    alphas, TRs = params[:nTR], params[nTR:]
    # simulate sequence
    jac, hes, num = epg.simulate(
        sequence(alphas, TRs, jac=True), 
        probe=[Jac, Hes, Num], 
        # callback=pruner, 
        max_nstate=nstate,
    )
    # CRLB
    cost, grad = stats.crlb(
        np.moveaxis(jac, -1, 0), 
        np.moveaxis(hes, -1, 0),
        W=weights,
        sigma2=sigma2,
        log=False,
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
    # return (1 - np.max(np.abs(np.diff(params[:nTR])))) * 10
    diff = np.diff(params, prepend=params[0])
    diff[nTR:] = 0
    return 1 - np.abs(diff)


# initial FA between 10 and 60
rstate = np.random.RandomState(0)
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
    "options": {"disp": True, "ftol": 1e-6, "maxiter": 250},
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
print(f"Results: alphas={res.x[:nTR]}, TRs={res.x[nTR:]}")
print(f"Optimization time: {duration/60:.1f} min")


#
# plot

fig, axes = plt.subplots(nrows=2, num="mrf-optim", sharex=True)
initconst = constraint_function(init).sum()
resconst = constraint_function(res.x).sum()

plt.sca(axes.flat[0])
plt.plot(init[:nTR], label=f"initial (constraint cost: {initconst:.2f})")
plt.plot(res.x[:nTR], label=f"optimized (constraint cost: {resconst:.2f})")
plt.legend()
plt.title("Flip angle values")
plt.ylabel("flip angle (degree)")
plt.ylim(0, 70)
plt.grid()
plt.sca(axes.flat[1])
plt.plot(init[nTR:], label="initial")
plt.plot(res.x[nTR:], label="optimized")
plt.legend()
plt.title("TR values")
plt.ylabel("TR (ms)")
plt.xlabel("echo index")
plt.ylim(10, 17)
plt.grid()
plt.tight_layout()


plt.figure("mrf-fingerprint")
sig0 = signal(init)
crb0 = costfun(init)
sig1 = signal(res.x)
crb1 = costfun(res.x)
plt.plot(np.abs(sig0[:, 0]), label=f"initial (log10(CRB): {crb0:0.2f})")
plt.plot(np.abs(sig1[:, 0]), label=f"optimized (log10(CRB): {crb1:0.2f})")
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
crb_tot = stats.crlb(jacs, W=weights, log=False, sigma2=sigma2)
crb_mag, crb_T1, crb_T2 = stats.crlb_split(jacs, W=weights, log=False)
plt.plot(crb_mag, ":", label="CRB magnitude")
plt.plot(crb_T1, ":", label="CRB T1")
plt.plot(crb_T2, ":", label="CRB T2")
plt.plot(crb_tot, label="CRB total")
plt.title("Evolution of CRB during optimization")
plt.xlabel("Iteration index")
plt.ylabel("CRB")
plt.legend()
plt.tight_layout()

plt.show()
