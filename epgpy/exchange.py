"""compartment exchange"""

import numpy as np
from . import common, operator

NAX = np.newaxis


#
# Exchange operator
class X(operator.Operator):
    """Exchange operator"""

    def __init__(
        self,
        tau,
        khi,
        *,
        axis=-1,
        T1=None,
        T2=None,
        g=None,
        name=None,
        duration=None,
    ):
        """Init exchange operator

        Args:
            tau: mixing time
            khi: float, exchange rate in 1/ms
                or N x N (first order) kinetic matrix
            T1: T1 of the compartments (ms)
            T2: T2 of the compartments (ms)
            g: chemical shift of the compartments (kHz)
        """
        params = common.map_arrays(tau=tau, T1=T1, T2=T2, g=g)

        if common.isscalar(khi):
            # if khi is scalar, assume 2 compartments
            khi = exchange_matrix(khi, axis=axis, ncomp=2)
        else:
            # if khi is array, assume formed kinetic matrix
            khi = np.asarray(khi)
            # check matrix
            if khi.ndim < 2:
                raise ValueError(f"Exchange matrix matrix must be at least 2D")
            elif khi.shape[:-1][axis] != khi.shape[-1]:
                raise ValueError(f"Exchange matrix must be square")
            elif not all(
                [
                    np.allclose(khi[..., i].sum(axis=axis), 0)
                    for i in range(khi.shape[-1])
                ]
            ):
                raise ValueError(f"Exchange matrix must sum to 0 along axis {axis}")

        # fix negative axis
        axis = int(khi.ndim + axis - 1) if axis < 0 else int(axis)

        # compute exchange operator
        mat = exchange_operator(tau, khi, axis=axis, T1=T1, T2=T2, g=g)

        self.axis = axis
        self.mat = mat

        self.khi = khi
        self.T1 = params["T1"]
        self.T2 = params["T2"]
        self.g = params["g"]
        self.tau = params["tau"]

        # special case: match duration and tau
        self._duration = duration
        if duration is True:
            duration = self.tau

        if name is None:  # default name
            name = common.repr_operator("X", ["tau", "khi"], [tau, khi])

        # init parent class
        super().__init__(name=name, duration=duration)

    @property
    def shape(self):
        return tuple(
            d for i, d in enumerate(self.mat.shape[:-1]) if i != (self.axis + 1)
        )

    def _apply(self, sm):
        """Apply echange and relaxation/precession"""
        xp = sm.array_module

        ax = self.axis
        ncomp = self.shape[ax]

        # check khi matrix for total magnetization
        if not xp.allclose(dotp(self.khi, sm.density[..., NAX], axes=[-1, ax]), 0):
            raise RuntimeError(
                "Exchange matrix `khi` does not conserve total magnetization"
            )

        # expand matrix
        dims = tuple(range(self.ndim + 1, sm.ndim + 2))
        mat = xp.expand_dims(self.mat, dims)

        if sm.shape[ax] == 1:
            # broadcast states
            sm.states = xp.concatenate([sm.states] * ncomp, axis=ax)
        elif sm.shape[ax] != ncomp:
            # should not happen
            raise RuntimeError("Invalid state matrix shape")

        # relaxation, precession & exchange
        sm.states = dotp(
            mat, insert_axis(sm.states - sm.equilibrium, ax), [ax + 1, ax + 1]
        )

        # add back equilibrium
        sm.states += sm.equilibrium
        return sm


#
# functions


def exchange_matrix(k, *, axis=-1, ncomp=2, densities=None):
    """convert scalar to 2d kinetic matrix

    Args:
        k: (n1 x ... x nk) array or float, exchange values
        axis: inserted echange axis
    Returns:
        arr: (n1 x ... x ncomp x ... x nk x ncomp)
        Kinetic matrix, with 2 new axes of size ncomp

    """
    k = np.asarray(k)
    if np.any(k < 0):
        raise ValueError("Cannot have negative echange rate")
    if axis > k.ndim:
        # insert missing dimensions if axis > 0
        dims = tuple(range(k.ndim, axis))
        k = np.expand_dims(k, dims)
    # fix negative axis
    axis = (k.ndim + axis + 1) if axis < 0 else axis
    # build kinetic matrix and move axis to final location
    kron = np.eye(ncomp) + (np.eye(ncomp) - 1) / (ncomp - 1)
    if densities is not None:
        kron /= densities
    return np.moveaxis(k[..., NAX, NAX] * kron, -2, axis)


def exchange_operator(tau, khi, *, axis=0, T1=None, T2=None, g=None):
    """Compute exchange operator

    References:
        Van Landeghem M, Haber A, D’espinose De Lacaillerie J-B, Blümich B
        Analysis of multisite 2D relaxation exchange NMR. Concepts Magn Reson 2010; 36A:153–169.

    """
    xp = common.get_array_module()

    khi = xp.asarray(khi)
    tau = xp.asarray(tau)
    T1 = xp.asarray(np.inf if T1 is None else T1)
    T2 = xp.asarray(np.inf if T2 is None else T2)
    g = xp.asarray(0 if g is None else g)

    # num compartments
    ncomp = khi.shape[-1]
    eye = xp.eye(ncomp)

    # common shape
    minshape = khi.shape[:-1]
    shape = broadcast_shapes(tau.shape, T1.shape, T2.shape, g.shape, minshape)
    ndim = len(shape)
    tau, T1, T2, g = expand_to([tau, T1, T2, g], ndim)
    T1, T2, g = broadcast_to([T1, T2, g], shape)

    # expand khi to match ndim
    dims = tuple(range(ndim - len(minshape)))
    khi = xp.expand_dims(khi, dims)

    # move exchange axis to the end
    tau, T1, T2, g = moveaxis_to([tau, T1, T2, g], axis, -1)

    # build evolution matrices
    xT = -khi + (-1 / T2 + 2j * np.pi * g)[..., NAX] * eye
    xL = -khi + (-1 / T1)[..., NAX] * eye

    # compute transition matrix
    mT = expm(xT * tau[..., NAX])
    mL = expm(xL * tau[..., NAX])

    # move back axis
    mT = xp.moveaxis(mT, (-2, -1), (axis, axis + 1))
    mL = xp.moveaxis(mL, (-2, -1), (axis, axis + 1))

    # stack matrices
    mat = xp.stack([mT, mT.conj(), mL], axis=-1)

    return mat


#
# utilities


def expand_to(arrs, ndim):
    """expand arrays"""
    xp = common.get_array_module(*arrs)
    expanded = []
    for arr in arrs:
        dims = tuple(range(arr.ndim, ndim))
        arr = xp.expand_dims(arr, dims)
        expanded.append(arr)
    return expanded


def broadcast_shapes(*shapes):
    return np.broadcast_shapes(*[shape[::-1] for shape in shapes])[::-1]


def broadcast_to(arrs, shape):
    """broadcast arrays"""
    xp = common.get_array_module(*arrs)
    broadcast = []
    for arr in arrs:
        arr = xp.broadcast_to(arr, shape)
        broadcast.append(arr)
    return broadcast


def moveaxis_to(arrs, ax1, ax2):
    xp = common.get_array_module(*arrs)
    moved = []
    for arr in arrs:
        moved.append(xp.moveaxis(arr, ax1, ax2))
    return moved


def insert_axis(arr, axis):
    xp = common.get_array_module(arr)
    return xp.expand_dims(arr, axis)


def dotp(a, b, axes=[-1, -1]):
    """dot product along custom axes"""
    xp = common.get_array_module()
    a, b = map(xp.asarray, [a, b])
    mov = xp.moveaxis
    return xp.einsum("...i,...i->...", mov(a, axes[0], -1), mov(b, axes[1], -1))


def transpose(mat):
    """transpose last 2 axes"""
    xp = common.get_array_module(mat)
    return xp.moveaxis(mat, -1, -2)


def expm(mat):
    """exponential of matrix"""
    tra = transpose
    xp = common.get_array_module(mat)
    matnorm = xp.linalg.norm(mat)
    if xp.isclose(matnorm, 0):
        return xp.eye(mat.shape[-1]).reshape(mat.shape)
    elif xp.allclose(mat, tra(mat).conj()):
        # hermician/symmetric
        evals, evecs = xp.linalg.eigh(mat / matnorm)
    else:
        # default
        evals, evecs = xp.linalg.eig(mat / matnorm)
    # eexp = xp.exp(evals * matnorm)
    eexp = xp.expm1(evals * matnorm) + 1  # more precise for small evals ?

    # (avoid using direct inversion)
    # ievecs = xp.linalg.inv(evecs)
    # return evecs @ (xp.exp(evals)[..., NAX] * ievecs)

    return tra(xp.linalg.solve(tra(evecs), eexp[..., NAX] * tra(evecs)))
