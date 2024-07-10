import numpy as np
from matplotlib import pyplot as plt
from epgpy import epg

# MSE sequence definition
"""
the `order1` keyword contains the list of parameters
for which to compute the signal's derivatives.

Available parameters for differentiation
epg.T(alpha, phi) --> `alpha`, `phi`
epg.E(tau, T1, T2, g) --> `tau`, `T1`, `T2`, `g`
epg.S(k) --> none (k is an integer)
"""

necho = 17
excit = epg.T(90, 90)
invert = epg.T(150, 0, order1="alpha")
relax = epg.E(4.5, 1400, 30, order1="T2")
shift = epg.S(1, duration=4.5)
adc = epg.ADC

# build sequence as a list
seq = [excit] + [shift, relax, invert, shift, relax, adc] * necho

# signal
times, signal = epg.simulate(seq, adc_time=True)

"""
The signal's derivatives are stored within the `gradient`
attribute of the state matrix object.
`sm.gradient` is a dict of state matrix derivatives
indexed by (<operator object>, <parameter name>) pairs.

An easy way to recover the Jacobian matrix
(the matrix of signal's derivatives),
is to use the `diff.Jacobian` probe operator.
"""

# signal jacobian
jprobe = epg.Jacobian(["alpha", 'T2'])
# retrieve the 17x2 Jacobian matrix
jac = epg.simulate(seq, probe=jprobe)


"""
To check the derivatives, one can compute the finite difference
approximation and compare.
"""

# compare with finite differences
eps = 1e-8
invert_alpha = epg.T(150 + eps, 0)
seq_alpha = [excit] + [shift, relax, invert_alpha, shift, relax, adc] * necho
fdiff_alpha = (epg.simulate(seq_alpha) - signal) / eps

relax_T2 = epg.E(4.5, 1400, 30 + eps)
seq_T2 = [excit] + [shift, relax_T2, invert, shift, relax_T2, adc] * necho
fdiff_T2 = (epg.simulate(seq_T2) - signal) / eps


# plot gradient
plt.figure("mse-diff")
hsignal = plt.plot(times, signal.real, "-", label="signal", color="gray")
plt.xlabel("time (ms)")
plt.ylabel("signal")
plt.title(r"MSE signal and its derivatives ($\alpha$ and $T2$)")
plt.twinx()
halpha = plt.plot(times, jac[:, 0].real)
halpha2 = plt.plot(times, fdiff_alpha.real, "+")
ht2 = plt.plot(times, jac[:, 1].real)
ht22 = plt.plot(times, fdiff_T2.real, "+")
plt.ylabel("signal derivative")
legend = plt.legend(
    [hsignal[0], halpha[0], halpha2[0], ht2[0], ht22[0]],
    [
        "signal ($S$)",
        r"$\partial S/\partial \alpha$",
        r"$f.diff(S, \alpha)$",
        r"$\partial S/\partial T2$",
        r"$f.diff(S, T2)$",
    ],
)
plt.tight_layout()
plt.show()


#
# 2nd derivatives

"""
The 2nd derivatives are also available.
They are activate by the hessian keyword in the operator's definition,
and stored in the `hessian` dictionary attribute of the state matrix.
"""

invert = epg.T(150, 0, order2="alpha")
relax = epg.E(4.5, 1400, 30, order2="T2")
seq = [excit] + [shift, relax, invert, shift, relax, adc] * necho

# retrieve the Hessian 17x2x2 tensor using the Hessian probe operator
hprobe = epg.Hessian(['alpha', 'T2'])
hessian = epg.simulate(seq, probe=hprobe)


""" Compare to finite difference approximate derivatives.
"""

# 1st order finite differences of signal's derivative
relax_T2 = epg.E(4.5, 1400, 30 + eps, order1="T2") # add order1 derivatives
seq_T2 = [excit] + [shift, relax_T2, invert, shift, relax_T2, adc] * necho
jprobe = epg.Jacobian('alpha')
fdiff_alpha_t2 = (epg.simulate(seq_T2, probe=jprobe)[:, 0] - jac[:, 0]) / eps

# plot 2nd derivative of signal w/r (T2, alpha)
plt.figure("mse-diff2")
plt.title(r"MSE: 2nd derivative ($\alpha$, $T2$) vs finite differences")
plt.plot(times, hessian[:, 0, 1].real, label=r"$\partial^2 S / \partial \alpha \partial T2$")
plt.plot(times, fdiff_alpha_t2.real, "+:", label='finite difference')
plt.xlabel("time (ms)")
plt.ylabel("2nd derivatives")
plt.legend()
plt.tight_layout()
plt.show()


#
# confidence intervals


from scipy import stats

"""
The signal's Jacobian matrix can be used for confidence interval calculation with the delta method

Note: for simplification, observation noise is wrongly assumed to be Gaussian instead of Rician

Ref:
https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html#confidence-and-prediction-intervals
"""

# add noise to signal (s.t. sse == 0.01)
noise = np.random.normal(size=necho)
noise *= np.sqrt(1e-2 / np.sum(noise**2))

obs = np.maximum(signal.real[:, 0] + noise, 0)
pred = signal.real[:, 0]

# sum of squared error 
sse = np.sum((pred - obs) ** 2)

# Jacobian and hessian
J = jac[..., 0]
H = hessian[..., 0]
nobs, nparam = J.shape

# number of degrees of freedom: num. echo - 2 (alpha and T2)
dof = nobs - nparam

# variance-covariance matrix
# V = np.linalg.inv((J.T @ J).real) # 1st order
V = np.linalg.inv((J.T @ J + H.T @ (pred - obs)).real) # 2nd order
V *= sse / dof
 
# c.int of reduced t statistics (mean=0, variance=1)
tval = np.asarray(stats.t.interval(0.95, dof))[1]

# confidence interval of alpha=150° and T2=30ms given above residuals
cint_alpha = np.sqrt(V[0, 0]) * tval
cint_T2 = np.sqrt(V[1, 1]) * tval

# confidence bands
predvar = np.einsum('np,pq,nq->n', J, V, J).real
cband = np.sqrt(predvar) * tval

# prediction bands
pband = np.sqrt(sse / dof + predvar) * tval

print(rf"c.int alpha: 150 +/- {cint_alpha}")
print(rf"c.int T2: 30 +/- {cint_T2}")

plt.figure("mse-cint")
plt.plot(times, obs, 'b+', label='observation')
plt.plot(times, pred, 'b', label='prediction')
plt.fill_between(
    times, pred - cband, pred + cband, alpha=0.3, label='95% confidence band'
)
plt.plot(times, pred - pband, "k:", label='95% prediction band')
plt.plot(times, pred + pband, "k:")

plt.title(
    f"MSE: confidence and prediction intervals\n"
    f"$cint(\\alpha)=150\\pm{cint_alpha:.1f}^\\circ$, "
    f"$cint(T2)=30\\pm{cint_T2:.1f}ms$"
)
plt.xlabel("time (ms)")
plt.ylabel("signal")
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()
