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
nTR = 50
T1, T2 = 1380, 80

order1_rf = [{f'alpha_{i}': {'alpha': 1}} for i in range(nTR)]
order2_rf = [{('T1', f'alpha_{i}'): {}, ('T2', f'alpha_{i}'): {}} for i in range(nTR)]

def rf(i, alpha, deriv=False):
    if not deriv:
        return epg.T(alpha, 90)
    return epg.T(alpha, 90, order1=order1_rf[i], order2=order2_rf[i])
    
order1_t12 = {'T1': {'T1': 1}, 'T2': {'T2': 1}}
order1_rlx = [{f'tau_{i}': {'tau': 1}, **order1_t12} for i in range(nTR)]
order2_rlx = [{('T1', f'tau_{i}'): {}, ('T2', f'tau_{i}'): {}} for i in range(nTR)]
def rlx(i, tau, deriv=False): 
    if not deriv:
        return epg.E(tau, T1, T2, order1=order1_t12, duration=True)
    return epg.E(
        tau, T1, T2, 
        order1=order1_rlx[i], order2=order2_rlx[i],
        duration=True,
    )
grd = epg.S(1)
adc = epg.ADC
seq = lambda alphas, taus, deriv: [
    [rf(i, alphas[i], deriv), rlx(i, taus[i], deriv), adc, grd] 
    for i in range(nTR)
]


""" optimize sequence parameters

# constraints
alpha in [10, 60]
TR in [11, 16]
|alpha_i - alpha_i+1| < 1

# W = diag(1, 1/T1^2, 1/T2^2)

"""


weights = [1, 1 / T1 ** 2, 1 / T2 ** 2]
params = [f'alpha_{i}' for i in range(nTR)] + [f'tau_{i}' for i in range(nTR)]
Jac = epg.Jacobian(['magnitude', 'T1', 'T2'])
Hes = epg.Hessian(['magnitude', 'T1', 'T2'], params)
def Num(sm):
    return (len(sm.order1), len(sm.order2))

pruner = diff.PartialsPruner(params, 1e-5)

def signal(params):
    alphas, taus = params[:nTR], params[nTR:]
    return epg.simulate(seq(alphas, taus, False))


def costfunction(params):
    """crlb cost function"""
    alphas, taus = params[:nTR], params[nTR:]
    jac = epg.simulate(seq(alphas, taus, False), probe=Jac)
    cost = stats.crlb(np.moveaxis(jac, -1, 0), W=weights, log=True)
    print(f"Cost function call: {cost}")
    # print(params)
    return cost


def costfunction_jac(params):
    """jacobian of cost function w/r to parameters alpha_xx and tau_xx"""
    alphas, taus = params[:nTR], params[nTR:]
    # breakpoint()
    jac, hes, num = epg.simulate(seq(alphas, taus, True), probe=[Jac, Hes, Num], callback=pruner)
    cost, grad = stats.crlb(
        np.moveaxis(jac, -1, 0), 
        np.moveaxis(hes, -1, 0),
        W=weights,
        log=True,
    )
    print(f"Cost function gradient call: {cost} (num partials: {max(num[:, 0]), max(num[:, 1])})")
    return grad.ravel()


def constraint_function(params):
    """constraint function: |alpha_i - alpha_i+1| < 1"""
    return (1 - np.max(np.abs(np.diff(params[:nTR])))) * 10


iterations = []


def callback(params):
    global iterations
    iterations.append({"x": params, "f(x)": costfunction(params)})


# initial values
init = list(np.sin(np.arange(nTR) * np.pi * 1.3 / nTR - np.pi / 2) * 5 + 35)
init += list(np.cos(np.arange(nTR) * np.pi * 5.0 / nTR) * 1 + 13.5)

# boundaries
bounds = [(10, 60)] * nTR + [(11, 16)] * nTR

config = {
    "method": "SLSQP",
    "jac": costfunction_jac,
    "bounds": bounds,
    "constraints": [{"type": "ineq", "fun": constraint_function}],
    "callback": callback,
    "options": {"disp": True, "ftol": 1e-8, "maxiter": 200},
}

tic = time.time()
res = optimize.minimize(costfunction, init, **config)
duration = time.time() - tic
print(f"Results: alphas={res.x[:nTR]}, taus={res.x[nTR:]}")
print(f"Optimization time: {duration}s")


initconst = constraint_function(init)
resconst = constraint_function(res.x)

fig, axes = plt.subplots(nrows=2, num="mrf-optim", sharex=True)
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
# plt.show()


varnames = [f"alpha{i}" for i in range(nTR)] + [f"tau{i}" for i in range(nTR)]
fig = plt.figure("mrf-fingerprint")
sig0 = signal(init)
crb0 = costfunction(init)
plt.plot(np.real(sig0[:, 0]), label=f"initial (log10(CRB): {crb0[0]:0.2f})")
sig1 = signal(res.x)
crb1 = costfunction(res.x)
plt.plot(np.real(sig1[:, 0]), label=f"optimized (log10(CRB): {crb1[0]:0.2f})")
plt.xlabel("echo index")
plt.ylabel("signal")
plt.title(f"Fingerprint for T1={T1}ms, T2={T2}ms")
plt.legend()
plt.grid()
# plt.show()


# plt.figure("mfr-iterations")
# jacs = [
#     seq.gradient(["magnitude", "T1", "T2"], **get_values(it["x"]))[1]
#     for it in iterations
# ]
# crb_tot = optim.crlb(jacs, W=weights, log=False)
# crb_mag, crb_T1, crb_T2 = optim.crlb_split(jacs, W=weights, log=False)
# plt.plot(crb_mag, ":", label="CRB magnitude")
# plt.plot(crb_T1, ":", label="CRB T1")
# plt.plot(crb_T2, ":", label="CRB T2")
# plt.plot(crb_tot, label="CRB total")
# plt.title("Evolution of CRB during optimization")
# plt.xlabel("Iteration index")
# plt.ylabel("log10(CRB)")
# plt.legend()
# plt.show()
