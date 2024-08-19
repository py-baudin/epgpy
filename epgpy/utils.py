import enum
import sys
import numpy as np
from . import common

NAX = np.newaxis

# constants
gamma_1H = 42.576 * 1e3  # kHz/T
gamma_23Na = 11.262 * 1e3  # kHz/T


def imaging(
    positions,
    states,
    wavenumbers,
    acctime=None,
    *,
    weights=None,
    modulation=None,
    voxel_shape="box",
    voxel_size=1,
    expand=True,
    reduce=True,
):
    """Discrete Fourier transform

    Args:
        states:         ... x nstate
        wavenumbers:    ... x nstate x ndim
        positions:      ... x ndim

    """

    xp = common.get_array_module(states)
    F = xp.asarray(states)
    k = xp.asarray(wavenumbers)
    t = xp.asarray(acctime) if acctime is not None else None
    pos = xp.asarray(positions)
    pos = pos if pos.ndim > 1 else pos[..., NAX]
    if expand:
        # insert pos dimensions into F and k
        dims = np.arange(pos.ndim - 1)
        F = xp.expand_dims(F, tuple(-2 - dims))
        k = xp.expand_dims(k, tuple(-3 - dims))
        if t is not None:
            t = xp.expand_dims(t, tuple(-2 - dims))

    # voxel shape
    if voxel_shape == "point":
        voxel = 1.0
    elif voxel_shape == "box":
        voxel = xp.sinc(k * voxel_size / 2 / np.pi).prod(-1)
        kmask = ~np.all(np.isclose(voxel, 0), axis=tuple(range(F.ndim - 1)))
        F, k, voxel = F[..., kmask], k[..., kmask, :], voxel[..., kmask]
        if t is not None:
            t = t[..., kmask]
    else:
        raise ValueError(f"Unknown voxel shape: {voxel_shape}")

    # modulation
    if t is not None and modulation is not None:
        modulation = xp.asarray(modulation)[..., NAX]
        amp, freq = modulation.real, modulation.imag
        mod = xp.exp(amp * xp.abs(t) + 2j * np.pi * t * freq)
        mmask = ~np.all(np.isclose(mod, 0), axis=tuple(range(F.ndim - 1)))
        F, k, mod = (
            F[..., mmask],
            k[
                ...,
                mmask,
            ],
            mod[..., mmask],
        )
    else:
        mod = 1.0

    # DFT
    kdim = min(k.shape[-1], pos.shape[-1])
    s = (
        voxel
        * mod
        * F
        * xp.exp(1j * xp.einsum("...ni,...i->...n", k[..., :kdim], pos[..., :kdim]))
    )

    # add up states
    s = s.sum(axis=-1)

    # weights
    if weights is not None:
        s *= xp.asarray(weights)

    # add up axes
    if reduce is True:
        return xp.sum(s)
    elif reduce is not False:
        return xp.sum(s, axis=reduce)
    return s


def dft(coords, states, wavenumbers, *, reduce=False):
    """simplified imaging function (discrete fourier transform)"""
    return imaging(coords, states, wavenumbers, reduce=reduce, voxel_shape="point")


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


def get_norm(states):
    xp = common.get_array_module()
    return xp.sqrt(xp.sum(xp.abs(states[..., 1:]) ** 2, axis=(-2, -1)))


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
