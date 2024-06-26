

#
# confidence intervals calculation
from scipy import stats

"""
The signal's Jacobian matrix can be used for confidence interval calculation.

Assumptions: the signal model's parameters are regressed from an acquired signal,
and the residual is i.i.d and normally distributed
"""

# sum of squared error (hypothesis)
sse = 1e-1

# Jacobian
J = jac[..., 0].real
nobs, nparam = J.shape

# number of degrees of freedom: num. echo - 2 (alpha and T2)
dof = nobs - nparam

# invert J.T @ J
V = np.linalg.inv(J.T @ J)

# c.int of reduced t statistics (mean=0, variance=1)
t = np.asarray(stats.t.interval(0.95, dof))

# confidence interval of alpha=150° and T2=30ms given above residuals
cint_alpha = sse / dof * np.sqrt(V[0, 0]) * t + 150
cint_T2 = sse / dof * np.sqrt(V[1, 1]) * t + 30

print(f"c.int alpha: {cint_alpha}")
print(f"c.int T2: {cint_T2}")

sig = signal.real[:, 0]
plt.figure("mse-cint")
plt.plot(times, sig)
plt.fill_between(
    times, sig - np.sqrt(sse / necho), sig + np.sqrt(sse / necho), alpha=0.3
)
plt.title(
    f"MSE signal with SSE={sse}\n"
    rf"c.int($\alpha$) : [{cint_alpha[0]:.1f}, {cint_alpha[1]:.1f}], "
    f"c.int(T2): [{cint_T2[0]:.1f}, {cint_T2[1]:.1f}]"
)
plt.xlabel("time (ms)")
plt.ylabel("signal")
plt.grid()
plt.show()
