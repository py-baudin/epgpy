"""Magnetization transfer operators and functions"""

import numpy as np
from . import utils

NAX = np.newaxis

""" 
# evolution operator with relaxation or precession rates
evo = R(tau, rL, rT)

# off-resonance saturation pulse for MT
W_off = saturation_rate(tau, rf, G_off)
sat = R(tau, [0, W_off])

# on-resonance saturation pulse
sat = R(tau, [0, W_on]) @ T([alpha, 0], phi)

"""


def saturation_rate(duration, rf, G, *, gamma=utils.gamma_1H):
    """Compute saturation rate of bound pool for a given rf pulse

    Validity domain: pulse's BW << bound pool BW

    Args
        duration: RF pulse duration (ms)
        rf: rf amplitude or waveform (uT)
        G: absorption line value of bound pool at off-resonance frequency (ms)
        gamma: (kHz/T)

    Returns:
        W: saturation rate (1/ms)

    > Graham SJ, Henkelman RM:
      Understanding pulsed magnetization transfer.
      Journal of Magnetic Resonance Imaging 1997; 7:903–912.

    """
    if np.isscalar(rf):
        # hard pulse
        integral = duration * rf**2
    else:
        integral = np.trapz(rf**2, dx=duration / (len(rf) - 1))

    W = np.pi * (1e-3 * 2 * np.pi * gamma) ** 2 * (1e-3 * G) * integral / duration
    return W * 1e-3


def absorption_rate(T2, lineshape, offres=0):
    """get absorption rate

    Args:
        T2: transverse relaxation (ms)
        offres: off-resonance frequency (kHz)
        lineshape: absorption lineshape
            'gaussian', 'lorentzian', 'super-lorentzian'
    Returns
        G: absoption rate (1/s)

    Refs.
    > Morrison C, Stanisz G, Henkelman RM:
      Modeling Magnetization Transfer for Biological-like Systems
      Using a Semi-solid Pool with a Super-Lorentzian Lineshape and Dipolar Reservoir.
      Journal of Magnetic Resonance, Series B 1995; 108:103–113.
    > Gloor M, Scheffler K, Bieri O:
      Quantitative magnetization transfer imaging using balanced SSFP.
      Magnetic Resonance in Medicine 2008; 60:691–700.


    """
    offres = np.asarray(offres)
    x = 2 * np.pi * T2 * offres

    if lineshape == "gaussian":
        G = T2 / (np.pi * 2) ** 0.5 * np.exp(-(x**2) / 2)

    elif lineshape == "lorentzian":
        G = T2 / np.pi * 1 / (1 + x**2)

    elif lineshape == "super-lorentzian":
        u = np.linspace(0, 1, 1000).reshape([1] * x.ndim + [-1])
        G = np.zeros(offres.shape)
        valid = np.abs(offres) >= 1
        # valid data points
        g = (
            1
            / np.abs(3 * u**2 - 1)
            * np.exp(-2 * (x[valid][..., NAX] / (3 * u**2 - 1)) ** 2)
        )
        G[valid] = T2 * (2 / np.pi) ** 0.5 * np.trapz(g, u, axis=-1)
        # extrapolated data points
        bounds = 2 * np.pi * T2 * np.array([1, 3, 5, 7, 9, 11])
        gref = (
            1
            / np.abs(3 * u**2 - 1)
            * np.exp(-2 * (bounds[..., NAX] / (3 * u**2 - 1)) ** 2)
        )
        Gref = T2 * (2 / np.pi) ** 0.5 * np.trapz(gref, u, axis=-1)
        G[~valid] = cubic_interp1d(
            x[~valid], np.r_[-bounds[::-1], bounds], np.r_[Gref[::-1], Gref]
        )

    else:
        raise ValueError(f"Unknown lineshape: {lineshape}")

    # return absorption rate in 1/s
    return G * 1e-3


def cubic_interp1d(x0, x, y):
    """
    Interpolate a 1-D function using cubic splines.
      x0 : a float or an 1d-array
      x : (N,) array_like
          A 1-D array of real/complex values.
      y : (N,) array_like
          A 1-D array of real values. The length of y along the
          interpolation axis must be equal to the length of x.

    Implement a trick to generate at first step the cholesky matrice L of
    the tridiagonal matrice A (thus L is a bidiagonal matrice that
    can be solved in two distinct loops).

    additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf
    """
    x = np.asfarray(x)
    y = np.asfarray(y)

    # remove non finite values
    # indexes = np.isfinite(x)
    # x = x[indexes]
    # y = y[indexes]

    # check if sorted
    if np.any(np.diff(x) < 0):
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

    size = len(x)

    xdiff = np.diff(x)
    ydiff = np.diff(y)

    # allocate buffer matrices
    Li = np.empty(size)
    Li_1 = np.empty(size - 1)
    z = np.empty(size)

    # fill diagonals Li and Li-1 and solve [L][y] = [B]
    Li[0] = np.sqrt(2 * xdiff[0])
    Li_1[0] = 0.0
    B0 = 0.0  # natural boundary
    z[0] = B0 / Li[0]

    for i in range(1, size - 1, 1):
        Li_1[i] = xdiff[i - 1] / Li[i - 1]
        Li[i] = np.sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
        Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    i = size - 1
    Li_1[i - 1] = xdiff[-1] / Li[i - 1]
    Li[i] = np.sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
    Bi = 0.0  # natural boundary
    z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

    # solve [L.T][x] = [y]
    i = size - 1
    z[i] = z[i] / Li[i]
    for i in range(size - 2, -1, -1):
        z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

    # find index
    index = x.searchsorted(x0)
    np.clip(index, 1, size - 1, index)

    xi1, xi0 = x[index], x[index - 1]
    yi1, yi0 = y[index], y[index - 1]
    zi1, zi0 = z[index], z[index - 1]
    hi1 = xi1 - xi0

    # calculate cubic
    f0 = (
        zi0 / (6 * hi1) * (xi1 - x0) ** 3
        + zi1 / (6 * hi1) * (x0 - xi0) ** 3
        + (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0)
        + (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
    )
    return f0
