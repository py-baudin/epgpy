""" Shift functions """

import numpy as np
from . import common, diff, utils

_S = slice(None)


class S(diff.DiffOperator):
    """Shift operator

    Methods: 'int-1d', 'int-nd', 'float-nd'
    """

    def __init__(
        self,
        k,
        *,
        nmax=None,
        kgrid=None,
        prune=1e-8,
        name=None,
        duration=None,
    ):
        """Init Shift operator

        Args:
            k: int, float (or array of)
                Units: rad/m
                Phase shifts increments or decrements
        """
        # check k
        if np.allclose(k, 0):
            raise TypeError("Cannot have k == 0")

        if not isinstance(k, int):
            k = np.atleast_2d(k)
            if not k.shape[-1] in [1, 2, 3, 4]:
                raise ValueError(f"k.shape[-1] must belong to [1, 2, 3, 4]")

        self.k = k
        self.nmax = nmax
        self.prune = prune
        self.kgrid = kgrid

        if not name:  # default name
            fmt = "" if isinstance(k, int) else ".2f"
            name = common.repr_operator("S", ["k"], [k], [fmt])

        # init parent class
        super().__init__(name=name, duration=duration)

    @property
    def nshift(self):
        if common.isscalar(self.k):
            return np.abs(self.k)
        # else
        return 1  # temp

    @property
    def shape(self):
        if common.isscalar(self.k):
            return (1,)
        return self.k.shape[:-1]

    @property
    def kdim(self):
        if common.isscalar(self.k):
            return 1
        return self.k.shape[-1]

    def copy(self, **kwargs):
        new = super().copy(**kwargs)
        new.k = self.k
        new.nmax = self.nmax
        new.prune = self.prune
        new.kgrid = self.kgrid
        return new

    def _apply(self, sm):
        """Shift states"""
        # check shift method
        method, shift = get_shift_method(self.k, sm.coords)

        if method == "int-1d":
            # basic 1d shift
            if sm.coords is not None:
                raise RuntimeError("Cannot use int-1d method on this state-matrix")

            # first add new state (if max_nstate not reached)
            nmax = sm.options.get("max_nstate") or self.nmax or np.inf
            sm.resize(min(sm.nstate + abs(shift), nmax))

            # shift states (inplace)
            sm.states = shift1d(sm.states, shift, inplace=True)

        elif method == "int-nd":
            # int nd-shift
            kdim = shift.shape[-1]
            if sm.coords is None or sm.kdim < kdim:
                sm.setup_coords(kdim)

            elif kdim < sm.kdim:
                diff = sm.kdim - kdim
                shift = np.pad(shift, [(0, 0)] * self.ndim + [(0, diff)])

            # apply (not inplace)
            opts = {
                "prune": bool(self.prune),
                "tol": self.prune,
                "nmax": self.nmax,
            }
            states, coords = shiftnd(sm.states, sm.coords, shift, **opts)
            nstate = (states.shape[-2] - 1) // 2
            sm.resize(nstate)
            sm.states, sm.coords = states, coords

        elif method == "float-nd":
            # float nd-shift-merge
            kdim = shift.shape[-1]
            if sm.coords is None or sm.kdim < kdim:
                sm.setup_coords(kdim)
            elif kdim < sm.kdim:
                diff = sm.kdim - kdim
                shift = np.pad(shift, [(0, 0)] * self.ndim + [(0, diff)])

            # kgrid
            kgrid = sm.options.get("kgrid") or self.kgrid
            if kgrid is None:
                raise AttributeError("kgrid not set")

            # apply (not inplace)
            coords = sm.coords * sm.ktvalue
            prune = sm.options.get("prune") or self.prune
            opts = {"prune": bool(prune), "tol": prune, "grid": kgrid}
            states, wavenums = shiftmerge(sm.states, coords, shift, **opts)
            nstate = (states.shape[-2] - 1) // 2
            sm.resize(nstate)
            sm.states = states
            sm.coords = wavenums / sm.ktvalue

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return sm

    def _derive1(self, sm, param):
        raise NotImplementedError()

    def _derive2(self, sm, param):
        raise NotImplementedError()

    # todo: def combine(self, ...)?


class G(S):
    """Gradient operator"""

    def __init__(self, tau, gradient, *, duration=None, **kwargs):
        """Setup shift operator using time `tau` (ms) and `gradient` (mT/m)"""

        tau, gradient = common.map_arrays([tau, gradient])

        if np.any(tau < 0):
            raise ValueError("Cannot have negative time")
        if not common.isscalar(gradient) and common.get_shape(gradient)[-1] > 3:
            raise ValueError("Only 3d gradients are allowed")

        # wavenumbers
        k = utils.get_wavenumber(tau, gradient)

        # duration
        duration = tau if duration is True else duration

        self.tau = tau
        self.gradient = gradient

        super().__init__(k, duration=duration, **kwargs)


class C(S):
    """Time-coherence operator"""

    def __init__(self, tau, *, duration=None, **kwargs):
        tau = common.map_arrays(tau)

        if np.any(tau < 0):
            raise ValueError("Cannot have negative time")

        # put time on fourth dimension
        k = np.stack([0 * tau] * 3 + [tau], axis=-1)

        # duration
        duration = tau if duration is True else duration

        self.tau = tau

        super().__init__(k, duration=duration, **kwargs)


#
# functions
def get_shift_method(k, coords):
    method = None
    if coords is None:
        shift = k
        if isinstance(k, int):
            method = "int-1d"
        elif isinstance(k.flat[0], np.integer):
            method = "int-nd"
        elif isinstance(k.flat[0], np.floating):
            method = "float-nd"

    elif np.issubdtype(coords.dtype, np.integer):
        # coords is int
        kdim = coords.shape[-1]
        if isinstance(k, int):
            shift = np.array([[int(k)] + [0] * (kdim - 1)])
            method = "int-nd"
        elif isinstance(k.flat[0], np.integer):
            shift = k
            method = "int-nd"
        elif isinstance(k.flat[0], np.floating):
            shift = k
            method = "float-nd"

    elif np.issubdtype(coords.dtype, np.floating):
        # coords is float
        method = "float-nd"
        kdim = coords.shape[-1]
        if isinstance(k, int):
            shift = np.array([[int(k)] + [0] * (kdim - 1)])
        else:
            shift = k

    if not method:
        raise ValueError("Unknown shift method")
    return method, shift


def get_nmax(shifts):
    """compute maximum cumulative phase-state"""
    nmax = 0
    cumshift = 0
    for shift in shifts:
        if not np.isscalar(shift):
            shift = np.asarray(shift)
        cumshift = cumshift + shift
        nmax = np.maximum(np.abs(cumshift), nmax)
    if np.isscalar(nmax):
        return nmax
    return nmax


def shift1d(states, n, *, inplace=False, nmax=None):
    """shift states by n"""
    if not inplace:
        ndim = max(states.ndim - 2, 0)
        nstate = (states.shape[-2] - 1) // 2
        diff = abs(n) if nmax is None else min(abs(n), nmax - nstate)
        if diff > 0:
            xp = common.get_array_module()
            states = xp.pad(states, [(0, 0)] * ndim + [(diff, diff), (0, 0)])
        elif diff < 0:
            states = states[..., diff:-diff, :]

    if n > 0:
        states[..., n:, 0] = states[..., :-n, 0]
        states[..., :-n, 1] = states[..., n:, 1]
        states[..., :n, 0] = 0
        states[..., -n:, 1] = 0
    else:
        states[..., :n, 0] = states[..., -n:, 0]
        states[..., -n:, 1] = states[..., :n, 1]
        states[..., n:, 0] = 0
        states[..., :-n, 1] = 0

    return states


def shiftnd(states, indices, shift, *, nmax=None, prune=True, tol=1e-8):
    """int 3d shift (not inplace)

    args:
        states: state matrix (... x nstate x 3)
        indices: phase-state indices (... x nstate x d)
        shift: phase-state increment (scalar or d-array)

        nmax: int (d-array of int) maximum phase-state index
            Above this value, phase-states are be cropped.
        prune: remove empty phase-states

    returns:
        states, indices: new phase-state matrix and indices

    """
    xp = common.get_array_module()

    sm = common.asarray(states)
    indices = common.asarray(indices, dtype=int)
    shift = common.expand_dims(common.asarray(shift), -2)

    # initial number of states
    n1 = sm.shape[-2]

    # add shift and take unique wavenumber indices
    kL = indices + 0 * shift
    k1T = kL + shift
    k2T = kL - shift
    k2, idx = unique_1d(xp.concatenate([kL, k1T, k2T], axis=-2), axis=-2)
    idxL, idxT = idx[:n1], idx[n1 : 2 * n1]
    keepL, keepT = slice(None), slice(None)

    if nmax is not None:
        # crop states if above a certain index
        keep = xp.any(
            xp.all(xp.abs(k2) <= nmax, axis=-1), axis=tuple(range(k2.ndim - 2))
        )
        if not xp.all(keep):
            k2 = k2[..., keep, :]

            mapidx = -xp.ones(keep.size, dtype=int)
            mapidx[keep] = xp.arange(k2.shape[-2])
            idxT, idxL = mapidx[idxT], mapidx[idxL]
            keepT, keepL = idxT >= 0, idxL >= 0

    # init new state matrix
    sm2 = xp.zeros(sm.shape[:-2] + (k2.shape[-2], 3), dtype=sm.dtype)

    # update location of L and T states
    sm2[..., idxL[keepL], 2] = sm[..., keepL, 2]
    sm2[..., idxT[keepT], 0] = sm[..., keepT, 0]
    sm2[..., 1] = sm2[..., ::-1, 0].conj()

    if prune:
        # prune empty phase-states
        nonzero = ~xp.all(
            xp.isclose(sm2, 0, atol=tol), axis=tuple(range(sm2.ndim - 2)) + (-1,)
        )
        nonzero[(k2.shape[-2] - 1) // 2] = True  # keep 0
        sm2 = sm2[..., nonzero, :]
        k2 = k2[..., nonzero, :]

    if k2.shape[-2] % 2 == 0:
        # should not happen
        raise ValueError(f"Asymmetrical state matrix")

    return sm2, k2


def shiftmerge(states, wavenums, shift, *, grid=1, prune=True, tol=1e-8):
    """nd shift-merge for arbitrary wavenumbers

    Args:
        states: state matrix (... x Nstate x 3)
        wavenums: wavenumbers (... x Nstate x kdim)
        shift: wavenumber increment (... x kdim)

        grid: gridsize (scalar or kdim)
        prune: remove empty phase-states

    returns
        states, wavenums: updated state matrix and wavenumbers

    Based on:
        Gao X, Kiselev VG, Lange T, Hennig J, Zaitsev M
        Three‐dimensional spatially resolved phase graph framework.
        Magn Reson Med 2021; 86:551–560.
    """
    xp = common.get_array_module()

    sm = common.asarray(states)
    wavenums = common.asarray(wavenums)
    shift = common.expand_dims(common.asarray(shift), -2)
    grid = grid * np.ones(wavenums.shape[-1])

    # initial number of states
    n1 = sm.shape[-2]

    # add shift
    # (prevent rounding issues due to numerical errors)
    kL = np.around(wavenums + 0 * shift, decimals=8)
    k1T = kL + shift
    k2T = kL - shift

    # quantize and take unique wavenumber indices
    # make sure q2 is symmetrical
    qL = xp.around(0.5 * (kL - kL[..., ::-1, :]) / grid).astype(int)
    q1T = xp.around(k1T / grid).astype(int)
    q2T = -q1T[..., ::-1, :]

    q2, idx = unique_1d(xp.concatenate([qL, q1T, q2T], axis=-2), axis=-2)
    idxL, idx1T, idx2T = idx[:n1], idx[n1 : 2 * n1], idx[2 * n1 :]

    # init new state matrix
    sm2 = xp.zeros(sm.shape[:-2] + (q2.shape[-2], 3), dtype=sm.dtype)

    # update L and T states:
    xp.add.at(sm2, (..., idxL, 2), sm[..., 2])
    xp.add.at(sm2, (..., idx1T, 0), sm[..., 0])
    sm2[..., 1] = sm2[..., ::-1, 0].conj()

    # xp.add.at(sm2, (..., idxL, 2), 0.5 * sm[..., 2])
    # xp.add.at(sm2, (..., idx1T, 0), 0.5 * sm[..., 0])
    # xp.add.at(sm2, (..., idx2T, 1), 0.5 * sm[..., 1])
    # reduce numerical errors by averaging
    # sm2 += sm2[..., ::-1, (1, 0, 2)].conj()

    # compute k2 as weigthed sum of merged wavenumbers
    w = xp.sum(xp.abs(sm), axis=tuple(i for i in range(sm.ndim - 2)), keepdims=True)
    wnorm = xp.zeros(q2.shape[:-1])
    xp.add.at(wnorm, (..., idxL), w[..., 2])
    xp.add.at(wnorm, (..., idx1T), w[..., 0])
    xp.add.at(wnorm, (..., idx2T), w[..., 1])

    k2 = xp.zeros(q2.shape, dtype=float)
    xp.add.at(k2, (..., idxL, _S), kL * w[..., 2:3])
    xp.add.at(k2, (..., idx1T, _S), k1T * w[..., 0:1])
    xp.add.at(k2, (..., idx2T, _S), k2T * w[..., 1:2])

    # find non-zero phase states
    nonzero = ~xp.all(
        xp.isclose(sm2, 0, atol=tol), axis=tuple(range(sm.ndim - 2)) + (-1,)
    )

    # normalize
    wnorm[..., ~nonzero] = 1.0
    k2 /= wnorm[..., np.newaxis]

    if prune:
        # prune empty phase-states
        nonzero[(k2.shape[-2] - 1) // 2] = True  # keep F0
        sm2 = sm2[..., nonzero, :]
        k2 = k2[..., nonzero, :]

    if k2.shape[-2] % 2 == 0:
        # should not happen
        raise ValueError(f"Asymmetrical state matrix")
    return sm2, k2


def unique_1d(values, axis=0):
    """faster unique, using python dictionary"""

    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("This function only works with integer arrays")

    xp = common.get_array_module()
    values = xp.moveaxis(values, axis, 0)
    shape = values.shape[1:]
    values = values.reshape(len(values), -1)

    # keep unique rows
    unique_set = {}

    # make rows into tuples
    # inverse = [unique_set.setdefault(tuple(row), len(unique_set)) for row in values.tolist()]

    # "hash" rows (faster)
    unique_set = {}
    vrange = np.ptp(values) + 1
    hash = np.dot(values - values.min(), [vrange**i for i in range(values.shape[1])])
    inverse = [unique_set.setdefault(row, len(unique_set)) for row in hash]

    # build `unique`array from inverse
    inverse = xp.array(inverse)
    unique = np.empty((len(unique_set), values.shape[1]), dtype=values.dtype)
    unique[inverse] = values

    # sort and return
    order = xp.lexsort(unique.T[::-1])
    unique = unique[order]
    unique = xp.moveaxis(unique.reshape((-1,) + shape), 0, axis)
    inverse = xp.argsort(order)[inverse]

    return unique, inverse
