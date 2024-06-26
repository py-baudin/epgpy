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
variables = [(invert, "alpha"), (relax, "T2")]
jprobe = diff.Jacobian(variables)
# retrieve the 17x2 Jacobian matrix
jac = epg.simulate(seq, probe=jprobe)


"""
To check the derivatives, one can compute the finite difference
approximation and compare.
"""

# compare with finite differences
eps = 1e-4
invert_ = diff.T(150 + eps, 0)
seq_alpha = [excit] + [shift, relax, invert_, shift, relax, adc] * necho
fdiff_alpha = (epg.simulate(seq_alpha) - signal) / eps

relax_ = diff.E(4.5, 1400, 30 + eps)
seq_T2 = [excit] + [shift, relax_, invert, shift, relax_, adc] * necho
fdiff_T2 = (epg.simulate(seq_T2) - signal) / eps


# plot gradient
plt.figure("mse-diff")
hsignal = plt.plot(times, signal.real, "-", label="signal", color="gray")
plt.xlabel("time (ms)")
plt.ylabel("signal")
plt.title("MSE signal and its derivatives")
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

invert = diff.T(150, 0, hessian=["alpha"])
relax = diff.E(4.5, 1400, 30, hessian=["T2"])
seq = [excit] + [shift, relax, invert, shift, relax, adc] * necho

# retrieve the Hessian 17x2x2 tensor using the Hessian probe operator
hprobe = diff.Hessian([(invert, "alpha"), (relax, "T2")])
hessian = epg.simulate(seq, probe=hprobe, squeeze=False)
# tmp: squeeze=False is required for now

""" Compare to finite difference approximate derivatives.

Note: 2nd order finite differences are less accurate than 1st order finite
differences of 1st derivative.
"""

# 1st order finite differences of signal's derivative
relax_ = diff.E(4.5, 1400, 30 + eps, gradient="T2")
seq_T2 = [excit] + [shift, relax_, invert, shift, relax_, adc] * necho
jprobe = diff.Jacobian([(invert, "alpha")])
fdiff_alpha_t2 = (epg.simulate(seq_T2, probe=jprobe)[:, 0] - jac[:, 0]) / eps

# 2nd order finite differences
seq_T2_alpha = [excit] + [shift, relax_, invert_, shift, relax_, adc] * necho
fdiff2_alpha_t2 = (
    (epg.simulate(seq_T2_alpha) - epg.simulate(seq_T2)) / eps
    - (epg.simulate(seq_alpha) - signal) / eps
) / eps


# plot 2nd derivative of signal w/r (T2, alpha)
plt.figure("mse-diff2")
# hsignal = plt.plot(times, signal.real, "-", label="signal", color="gray")
# plt.ylabel("signal")
plt.title("MSE: 2nd derivative (alpha, T2) vs finite differences")
# halpha_T2 = plt.plot(times, hessian[:, 0, 1].real)
# halpha_T2_fdiff1 = plt.plot(times, fdiff_alpha_t2.real, "+:")
# halpha_T2_fdiff2 = plt.plot(times, f2diff_alpha_t2.real, "+:")
plt.xlabel("time (ms)")
plt.ylabel("2nd derivatives - finite differences")
# plt.twinx()
# halpha_T2 = plt.plot(times, hessian[:, 0, 1].real - hessian[:, 0, 1].real )
halpha_T2_fdiff1 = plt.plot(times, hessian[:, 0, 1].real - fdiff_alpha_t2.real, "+:")
halpha_T2_fdiff2 = plt.plot(times, hessian[:, 0, 1].real - fdiff2_alpha_t2.real, "+:")
legend = plt.legend(
    [halpha_T2_fdiff1[0], halpha_T2_fdiff2[0]],
    [
        r"$\partial^2 S/\partial \alpha \partial T2 - f.diff(\partial S/\partial \alpha, T2)$",
        r"$\partial^2 S/\partial \alpha \partial T2 - f.diff(f.diff(S, T2), \alpha)$",
    ],
)
plt.tight_layout()
plt.show()
