"""inverse Laplace Transform

cf.
> Fricke SN, Seymour JD, Battistel MD, Freedberg DI, Eads CD, Augustine MP:
  Data processing in NMR relaxometry using the matrix pencil.
  Journal of Magnetic Resonance 2020; 313:106704.

> Eads CD: Analysis of multicomponent exponential magnetic resonance relaxation data:
  automatable parameterization and interpretable display protocols.



"""

import numpy as np
from scipy import optimize

NAX = np.newaxis


def get_bounds(times, tol=5e-1):
    """estimate bounds that can be reached given time vector"""
    mintime = np.min(np.diff(times))
    maxtime = np.ptp(times)
    minb = -np.log(1 - tol) / maxtime
    maxb = -np.log(tol) / mintime
    # maxb = 1/mintime
    return minb, maxb


def get_kernel(times, bounds, num):
    """generate time-rate exponential kernel"""
    times = np.asarray(times)
    rates = np.geomspace(bounds[0], bounds[1], num)
    kernel = np.exp(-np.outer(times, rates))
    return rates, kernel


def get_resolution(times, bounds, *, tol=1e-3, ncurve=100):
    """compute optimal resolution given bounds and tolerance"""
    rates = np.geomspace(bounds[0], bounds[1], ncurve)
    y = np.exp(-np.outer(times, rates))
    err = np.inf
    num = 2
    # print(f'bounds={bounds}')
    while True:
        # get kernel for this number of increments
        rates, K = get_kernel(times, bounds, num)
        res = rates[1] / rates[0]
        # get maximum tolerable error given kernel
        sopt, *_ = np.linalg.lstsq(K.T @ K, K.T @ y, rcond=None)
        err = np.linalg.norm(K @ sopt - y, axis=0).max()
        # err = np.percentile(np.linalg.norm(K @ sopt - y, axis=0), 95)
        # print(f'num: {num}, res: {res}, err: {err}')
        if err < tol:
            break
        num += 1
    return res, num


#
# ILT / ARM


def tsvd(M, tol=1e-5):
    """truncated svd"""
    u, d, v = np.linalg.svd(M)
    # keep = max(np.argmax(np.cumsum(d) / np.sum(d) > 1 - tol), 1)
    khi2 = (
        np.array([np.sum((M - (u[:, :k] * d[:k]) @ v[:k]) ** 2) for k in range(len(d))])
        / M.size
    )
    keep = np.argmax(khi2 < tol)
    return u[:, :keep], d[:keep], v[:keep]


def ilt1d(times, signal, *, bounds=None, kernel=None, ls=True):
    """ILT with MPM model"""
    times = np.asarray(times)
    sig = np.asarray(signal)
    if times.size != sig.shape[0]:
        raise ValueError(sig)
    if np.ptp(np.diff(times)) > 1e-8:
        raise ValueError("Non-regular time sampling")
    dt = times[1] - times[0]

    # get bounds:
    bounds = bounds or get_bounds(times)

    if kernel is None:
        # get resolution
        res, num = get_resolution(times, bounds)
        # get kernel
        _, kernel = get_kernel(times, bounds, num)

    # build auto-regressive matrices
    Nt, Nr = kernel.shape
    Y1 = np.stack([sig[i : i + Nt // 2] for i in range(Nt // 2)], axis=1)
    Y2 = np.stack([sig[i + 1 : i + Nt // 2 + 1] for i in range(Nt // 2)], axis=1)

    # pseudo inverse of Y1 with truncated SVD
    U, d, V = tsvd(Y1)
    p = len(d)
    # rates
    zs = np.linalg.eigvals((1 / d[:, NAX] * U.T) @ Y2 @ V.T)

    # filter bad components
    minz = np.exp(-dt * bounds[1])
    maxz = np.exp(-bounds[0] * dt)
    keep = np.isclose(zs.imag, 0) & (zs.real >= minz) & (zs.real <= maxz)
    if keep.sum():
        zs = np.sort(zs[keep].real)[:p]
    else:
        zs = np.max(zs.real)[NAX]

    # get relaxation rates
    r = -np.log(np.abs(zs)) / dt

    if ls:
        # LS refinement
        r, a = ilt1d_ls(times, signal, r)
    else:
        # amplitudes
        Z = np.linalg.pinv(zs[:, NAX] ** np.arange(Nt // 2)).T
        A = Z @ Y2 @ Z.T
        a = np.diag(A)

    keep = a > 0
    r, a = r[keep], a[keep]
    return r, a


def ilt1d_ls(times, signal, rates):
    """ILT, LS refinement"""
    t = np.asarray(times)
    y = np.asarray(signal)
    y2 = np.dot(y, y)

    def cost(r):  # cost function on rates
        # R = np.pow(r, -t[:, NAX])
        R = np.exp(-np.outer(t, r))
        Ry = R.T @ y
        cost = y2 - Ry.T @ np.linalg.solve(np.dot(R.T, R), Ry)
        return cost

    # transpose tensor
    tr = lambda A: np.moveaxis(A, 0, 1)
    mult = lambda A, B: np.einsum("ij...,jk...->ik...", A, B)

    def jac(r):
        R = np.exp(-np.outer(t, r))
        dr = np.diag(r)
        dR = -(t * np.exp(-t * dr[NAX].T)).T * np.eye(len(r))  # Nt x Nr x Nr
        x = np.linalg.solve(R.T @ R, R.T @ y[:, NAX])  # inv(R'R)) R'y
        dRR = mult(tr(dR), R) + mult(tr(R), dR)  # dR'R + R'dR
        dcost = mult(x.T, -2 * mult(tr(dR), y[:, NAX]) + mult(dRR, x))
        return dcost.ravel()

    bounds = [(0, None)] * len(rates)
    res = optimize.minimize(
        cost, rates, jac=jac, bounds=bounds
    )  # tions={'disp': True})
    r = res.x

    # amplitudes
    R = np.exp(-np.outer(t, r))
    a = np.linalg.solve(R.T @ R, R.T @ y)
    nonzero = (r > 1e-8) & (a > 1e-8)
    return r[nonzero], a[nonzero]


def flt1d(times, rates, amplitudes):
    """forward laplace transform"""
    r = np.asarray(rates)
    a = np.asarray(amplitudes)
    t = np.asarray(times)
    return np.sum(a * np.exp(-np.outer(t, r)), axis=1)


def ilt1d_crb(times, signal, rates, amps):
    """CRLB of MPM model"""
    times, signal, rates, amps = map(np.asarray, [times, signal, rates, amps])

    m = len(times) // 2
    n = len(rates)
    Y = np.stack([signal[i : i + m] for i in range(m)], axis=1)
    # Y = np.c_[signal[:m]]

    # Xhi2 statistic
    dt = times[1] - times[0]
    IJ = np.stack([np.arange(j, j + m) for j in range(m)], 1)
    # IJ = np.c_[np.arange(m)]
    Z = np.exp(-dt * rates[:, NAX, NAX] * IJ)
    D = amps[:, NAX, NAX] * Z
    xi2 = np.sum((Y - D.sum(0)) ** 2)  # / (m - 1)**2

    ## gradient matrix
    dxida = -2 * np.sum(Z * (Y - D.sum(0)), axis=(1, 2))  # / (m-1)**2
    dxidr = 2 * np.sum((dt * IJ * D) * (Y - D.sum(0)), axis=(1, 2))  # / (m-1)**2
    dxi = np.concatenate([dxidr, dxida], 0)
    # H = np.outer(dxi, dxi)

    ## hessian matrix
    tprod = lambda A, B: np.sum(A[:, NAX] * B, axis=(-2, -1))
    dprod = lambda A, B: np.sum(
        A[:, NAX] * B * np.eye(n)[:, :, NAX, NAX], axis=(-2, -1)
    )
    dxida2 = 2 * tprod(Z, Z)
    dxidrda = 2 * tprod(dt * IJ * D, -Z)
    dxidrda += 2 * dprod(dt * IJ * Z, (Y - D.sum(0))[NAX])

    dxidr2 = 2 * tprod(dt * IJ * D, dt * IJ * D)
    dxidr2 += -2 * dprod(dt**2 * IJ**2 * D, (Y - D.sum(0))[NAX])
    # H = np.block([[dxidr2, dxidrda], [dxidrda.T, dxida2]]) #/ (m-1)**2
    H = dxidr2

    # # tmp: check with finite difference
    # fZ = lambda r: np.exp(-dt * r[:, NAX, NAX] * IJ)
    # fD = lambda a, r: a[:, NAX, NAX] * fZ(r)
    # fxi2 = lambda a, r: np.sum((Y - fD(a, r).sum(0)) ** 2)  # / (m - 1)**2
    # fdxida = lambda a, r: -2 * np.sum(
    #     fZ(r) * (Y - fD(a, r).sum(0)), axis=(1, 2)
    # )  # / (m-1)**2
    # fdxidr = lambda a, r: 2 * np.sum(
    #     (dt * IJ * fD(a, r)) * (Y - fD(a, r).sum(0)), axis=(1, 2)
    # )  # / (m-1)**2
    # dr1 = np.eye(n)[0] * 1j * 1e-5
    # da1 = np.eye(n)[0] * 1j * 1e-5
    # # dr2 = np.eye(n)[1]*1j*1e-5
    # # da2 = np.eye(n)[1]*1j*1e-5

    # Fisher information matrix and CRB
    I = np.linalg.pinv(H / xi2)
    crb = np.diag(I) * 1
    # print(f"xi2: {xi2:.3}, r: {r}, crb_r: {crb[:n]}, a: {a}, crb_a:{crb[n:]}")
    crb[np.isnan(crb) | (crb < 0) | (crb > 1e3)] = 0

    # return crb[:n], crb[n:]
    return crb[:n], np.zeros(n)


def qcr(bounds, r, a, widths, *, num=None):
    """quasi continuous rendering"""

    # render resolution
    num = num or 1000
    rates = np.geomspace(bounds[0], bounds[1], num)
    logrates = np.log(rates)
    logres = logrates[1] - logrates[0]

    widths = [widths] * len(r) if np.isscalar(widths) else widths

    # convolve with Gaussian function
    render = 0
    for i in range(len(r)):
        # spikes
        spikes = np.zeros(num)
        spikes[np.digitize(np.log(r[i]), logrates) - 1] = a[i]
        # gaussian function
        if (widths[i] <= 0) or (np.log(widths[i]) < logres * 3):
            # if too narrow, skip convolution
            render = render + spikes
            continue
        sigma = np.log(widths[i])
        nconv = int(5 * sigma / logres + 0.5)
        xvals = np.arange(-nconv, nconv + 1) * logres
        gauss = np.exp(-0.5 * xvals**2 / sigma**2)
        # convolve spikes and gaussian and trim render to bounds
        render_i = np.convolve(spikes, gauss, mode="full")
        render = render + render_i[nconv:-nconv]
    return logrates, render
