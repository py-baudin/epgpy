""" Linear operators and functions """
import abc
import numpy as np
from . import operator, common

NAX = np.newaxis


class LinearOperator(operator.Operator, abc.ABC):
    """Base class for linear operators"""

    @abc.abstractmethod
    def __matmul__(self, other):
        pass


class ScalarOp(LinearOperator):
    """State-wise scalar multiplication operator"""

    def __init__(self, coeff, coeff0=None, *, axes=None, **kwargs):
        """Initialize operator"""
        coeff = scalar_op(coeff)
        if coeff0 is not None:
            coeff0 = scalar_op(coeff0)
            coeff, coeff0 = np.broadcast_arrays(coeff, coeff0)

        if axes is not None:
            coeff = set_axes(1, coeff, axes)
            coeff0 = None if coeff0 is None else set_axes(1, coeff0, axes)

        self.coeff = coeff
        self.coeff0 = coeff0
        super().__init__(**kwargs)

    @property
    def shape(self):
        return self.coeff.shape[:-1]

    @property
    def mat(self):
        """return diagonal matrix of coefficients"""
        xp = common.get_array_module(self.coeff)
        return self.coeff[..., NAX] * xp.eye(3)

    @property
    def mat0(self):
        """return diagonal matrix of coefficients"""
        if self.coeff0 is None:
            return None
        xp = common.get_array_module(self.coeff)
        return self.coeff0[..., NAX] * xp.eye(3)

    def __matmul__(self, other):
        """multiply operators"""
        if not isinstance(other, ScalarOp):
            return NotImplemented
        coeff, coeff0 = scalar_combine(
            self.coeff, other.coeff, self.coeff0, other.coeff0
        )
        name = self.name + "|" + other.name
        duration = self.duration + other.duration
        return ScalarOp(coeff, coeff0, name=name, duration=duration)

    def _apply(self, sm):
        """apply inplace"""
        sm.states = scalar_prod(self.coeff, sm.states, inplace=True)
        if self.coeff0 is not None:
            sm.states += scalar_prod(self.coeff0, sm.equilibrium, inplace=False)
        return sm


class MatrixOp(LinearOperator):
    """state-wise matrix multiplication operator"""

    def __init__(self, mat, mat0=None, *, axes=None, **kwargs):
        """Initialize operator

        Args:
            mat: sequence of 3x3 matrices to multiply with state matrix
            mat0: sequence of 3x3 matrices to multiply with equilibrium matrix
            axes: shift axes to given indices

        """
        # setup matrix operator
        mat = matrix_op(mat)
        if mat0 is not None:
            mat0 = matrix_op(mat0)
            mat, mat0 = np.broadcast_arrays(mat, mat0)

        # axes
        if axes is not None:
            mat = set_axes(2, mat, axes)
            mat0 = None if mat0 is None else set_axes(2, mat0, axes)

        self.mat = common.asarray(mat)
        self.mat0 = None if mat0 is None else common.asarray(mat0)

        # init parent class
        super().__init__(**kwargs)

    @property
    def shape(self):
        return self.mat.shape[:-2]

    def __matmul__(self, other):
        """multiply operators"""
        if not isinstance(other, (MatrixOp, ScalarOp)):
            return NotImplemented
        mat, mat0 = matrix_combine(self.mat, other.mat, self.mat0, other.mat0)
        name = self.name + "|" + other.name
        duration = self.duration + other.duration
        return MatrixOp(mat, mat0, name=name, duration=duration)

    def __rmatmul__(self, other):
        """multiply operators"""
        if not isinstance(other, (MatrixOp, ScalarOp)):
            return NotImplemented
        mat, mat0 = matrix_combine(other.mat, self.mat, other.mat0, self.mat0)
        name = other.name + self.name
        duration = self.duration + other.duration
        return MatrixOp(mat, mat0, name=name, duration=duration)

    def _apply(self, sm):
        """Apply transform to state matrix (inplace)"""
        sm.states = matrix_prod(self.mat, sm.states, inplace=True)
        if self.mat0 is not None:
            sm.states += matrix_prod(self.mat0, sm.equilibrium, inplace=False)
        return sm


#
# functions


def scalar_op(arr, check=True):
    """setup scalar operator"""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[NAX]

    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected ...x3 array shape, found: {arr.shape}")
    elif not check:
        return arr
    elif not np.allclose(arr, arr[..., (1, 0, 2)].conj()):
        raise ValueError(f"Invalid coefficients: {arr}")

    xp = common.get_array_module()
    return xp.asarray(arr)


def scalar_combine(coeff1, coeff2, coeff01, coeff02):
    """combine 2 scalar operators"""
    xp = common.get_array_module(coeff1, coeff2)
    coeff1, coeff2, coeff01, coeff02 = extend_operators(
        1, coeff1, coeff2, coeff01, coeff02
    )
    coeff = coeff2 * coeff1
    if coeff01 is None and coeff02 is None:
        coeff0 = None
    elif coeff01 is None:
        coeff0 = coeff02
    else:
        coeff0 = coeff2 * coeff01
        if coeff02 is not None:
            coeff0 += coeff02
    return coeff, coeff0


def scalar_prod(coeff, states, *, inplace=False):
    """element-wise product product"""
    xp = common.get_array_module(coeff, states)

    # expand mat dims if needed
    dims = tuple(range(coeff.ndim - 1, states.ndim - 1))
    coeff = xp.expand_dims(coeff, dims)

    # broadcastable
    broadcastable = all(s1 <= s2 for s1, s2 in zip(coeff.shape[:-1], states.shape[:-2]))

    # multiply
    if inplace and broadcastable:
        states *= coeff
    else:
        states = states * coeff
    return states


def matrix_op(mat, check=True):
    """setup matrix operator"""
    mat = np.asarray(mat)
    if mat.ndim == 2:
        mat = mat[NAX]
    if mat.ndim < 3 or mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ...x3x3 array shape, found: {mat.shape}")
    elif not check:
        return mat
    elif not np.allclose(mat, mat[..., (1, 0, 2), :][..., (1, 0, 2)].conj()):
        raise ValueError(f"Invalid matrix coefficients: {mat}")

    xp = common.get_array_module()
    return xp.asarray(mat)


def matrix_combine(mat1, mat2, mat01, mat02):
    """combine 2 matrix operators"""
    xp = common.get_array_module(mat1, mat2)
    mat1, mat2, mat01, mat02 = extend_operators(2, mat1, mat2, mat01, mat02)
    mat = xp.einsum("...ij,...jk->...ik", mat2, mat1)
    if mat01 is None and mat02 is None:
        mat0 = None
    elif mat01 is None:
        mat0 = mat02
    else:
        mat0 = xp.einsum("...ij,...jk->...ik", mat2, mat01)
        if mat02 is not None:
            mat0 += mat02
    return mat, mat0


def matrix_combine_multi(mats):
    """temp"""
    xp = common.get_array_module(mats[0])
    mat = mats[0]
    for mat_ in mats[1:]:
        mat = xp.einsum("...ij,...jk->...ik", mat_, mat)
    return mat


def matrix_prod(mat, states, *, inplace=False):
    """matrix multiplication"""
    xp = common.get_array_module(mat, states)

    # expand mat dims if needed
    dims = tuple(range(mat.ndim - 2, states.ndim - 1))
    mat = xp.expand_dims(mat, dims)

    # broadcastable
    broadcastable = all(s1 <= s2 for s1, s2 in zip(mat.shape[:-2], states.shape[:-2]))

    # use inplace mult only with numpy
    inplace = inplace & (xp.__name__ == "numpy") & broadcastable

    if inplace:
        return xp.einsum("...ij,...j->...i", mat, states, out=states)
    else:
        return xp.matmul(mat, states[..., xp.newaxis])[..., 0]


def set_axes(nd, mats, axes):
    ndim = mats.ndim - nd
    if isinstance(axes, int):
        axes = tuple(range(axes, axes + ndim))
    elif not isinstance(axes, tuple) or not all(isinstance(ax, int) for ax in axes):
        raise ValueError(f"Invalid axes: {axes}")

    # expand dimensions
    newdims = tuple([i for i in range(max(axes)) if not i in axes])
    mats = common.expand_dims(mats, newdims)
    return mats


def extend_operators(nd, *ops):
    """extend operators to make them broadcastable"""
    shapes = [common.get_shape(op)[:-nd] for op in ops]
    shape = common.broadcast_shapes(*shapes, append=True)
    ndim = len(shape)
    extended = []
    for op in ops:
        if op is None:
            extended.append(None)
        else:
            dims = tuple(range(op.ndim - nd, ndim))
            extended.append(common.expand_dims(op, dims))
    return extended
