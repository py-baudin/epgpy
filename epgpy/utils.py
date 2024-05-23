import enum
import sys
import numpy as np
from . import common

# constants
gamma_1H = 42.576 * 1e3  # kHz/T
gamma_23Na = 11.262 * 1e3  # kHz/T


def check_states(states):
    """check state matrix validity"""
    xp = common.get_array_module(states)
    return xp.allclose(states, states[..., ::-1, [1, 0, 2]].conj())


# axes
def Axes(*names):
    """helper function to create Enum for axes
    ex:
    ```
        # create axes Enum
        axes = Axes("T2", "B1")

        # use axes enum in place of indices
        sm.shape[axes.T2]
    ```
    """
    return enum.IntEnum("Axes", names, start=0)


#
# conversion functions


def get_wavenumber(grad, duration, gamma=gamma_1H):
    """compute wavenumber resulting from gradient application

    Args:
        grad: gradient strength (mT/m)
        duration: gradient duration (ms)
        gamma: gyromagnetic ratio (kHz/T)

    Returns
        wavenumber (rad/m)

    """
    return 2 * np.pi * gamma * np.asarray(grad) * 1e-3 * np.asarray(duration)


# profiles and utilities


def spatial_range(fov, nvalue=100):
    """make array of spatial values

    Parameters:
    ===
        fov: size of the spatial range (in mm)
        nvalue: number of values of array
    """
    return fov * np.linspace(-0.5, 0.5, nvalue)


def space_to_freq(grad, positions, *, gamma=gamma_1H):
    """Convert a (array of) spatial locations into frequencies

       Useful for profile simulation

    Parameters
    ===
        gradient: float
            Gradient value in mT/m
        positions: float or array of floats
            Relative spatial locations in mm
        gamma: float
            Gyromagnetic ratio in kHz/T (default: value for ^1H)

    Returns
    ===
        frequencies: array of float
            Frequency array in kHz

    """
    if not np.isscalar(positions):
        positions = np.asarray(positions)
    return grad * 1e-6 * gamma * positions


def freq_to_space(grad, frequencies, *, gamma=gamma_1H):
    """reverse of the above"""
    return frequencies / grad / gamma * 1e6


# misc


def progressbar(it, prefix="", size=60, out=sys.stdout):
    """https://stackoverflow.com/questions/3160699/python-progress-bar"""
    count = len(it)

    def show(j):
        x = int(size * j / count)
        print(
            "{}[{}{}] {}/{}".format(prefix, "#" * x, "." * (size - x), j, count),
            end="\r",
            file=out,
            flush=True,
        )

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i + 1)
    print("\n", flush=True, file=out)


class DeferredGetter(dict):
    """dict for lazy evaluating object getters"""

    def __init__(self, obj, getters):
        self._obj = obj
        self._getters = getters
        for getter in getters:
            self[getter] = None

    def __getitem__(self, item):
        if item in self._getters:
            return getattr(self._obj, item)
        return dict.__getitem__(self, item)

    def __getattr__(self, item):
        if item in self._getters:
            return self[item]
        super().__getattr__(item)
