""" Scalar operator and functions """
import numpy as np
from . import common, oplinear, opscalar

NAX = np.newaxis


class MatrixOp(oplinear.LinearOperator):
    """state-wise matrix multiplication operator"""

    def __init__(self, mat, mat0=None, *, axes=None, **kwargs):
        """Initialize operator

        Args:
            mat: sequence of 3x3 matrices to multiply with state matrix
            mat0: sequence of 3x3 matrices to multiply with equilibrium matrix
            axes: shift axes to given indices

        """
        # setup matrix operator
        self.mat, self.mat0 = matrix_setup(mat, mat0, axes=axes)

        # derivatives
        self.dmat, self.dmat0 = None, None
        dmat, dmat0 = kwargs.pop('dmat', None), kwargs.pop('dmat0', None)
        if dmat is not None:
            np.broadcast_shapes(mat.shape, dmat.shape)
            self.dmat, self.dmat0 = matrix_setup(dmat, dmat0, axes=axes)

        self.d2mat, self.d2mat0 = None, None
        d2mat, d2mat0 = kwargs.pop('d2mat', None), kwargs.pop('d2mat0', None)
        if d2mat is not None:
            np.broadcast_shapes(mat.shape, d2mat.shape)
            self.d2mat, self.d2mat0 = matrix_setup(d2mat, d2mat0, axes=axes)
    
        # init parent class
        super().__init__(**kwargs)

    @property
    def shape(self):
        return self.mat.shape[:-2]

    def __matmul__(self, other):
        """multiply operators"""
        if not isinstance(other, (MatrixOp, opscalar.ScalarOp)):
            return NotImplemented
        return self.combine([self, other])

    def __rmatmul__(self, other):
        """multiply operators"""
        if not isinstance(other, (MatrixOp, opscalar.ScalarOp)):
            return NotImplemented
        return self.combine([other, self])

    def _apply(self, sm):
        """Apply transform to state matrix (inplace)"""
        return matrix_apply(self.mat, self.mat0, sm)

    def _derive1(self, sm, param):
        if self.dmat is None:
            raise NotImplementedError('derive1')
        return matrix_apply(self.dmat, self.dmat0, sm)

    def _derive2(self, sm, params):
        if self.d2mat is None:
            raise NotImplementedError('derive2')
        return matrix_apply(self.d2mat, self.d2mat0, sm)
    
    @staticmethod
    def combine(ops, name=None):
        """ combine multiple scalar operators"""
        mat, mat0 = ops[0].mat, ops[0].mat0
        dmat, dmat0 = ops[0].dmat, ops[0].dmat0
        d2mat, d2mat0 = ops[0].d2mat, ops[0].d2mat0
        for op in ops[1:]:
            if not isinstance(op, (MatrixOp, opscalar.ScalarOp)):
                raise NotImplementedError()
            mat, mat0 = matrix_combine(mat, op.mat, mat0, op.mat0)
            if dmat is not None:
                dmat, dmat0 = matrix_combine(dmat, op.dmat, dmat0, op.dmat0)
            if d2mat is not None:
                d2mat, d2mat0 = matrix_combine(d2mat, op.d2mat, d2mat0, op.d2mat0)

        if name is None:
            name = "|".join(op.name for op in ops)
        duration = sum(op.duration for op in ops)
        
        return MatrixOp(mat, mat0, dmat=dmat, dmat0=dmat0, d2mat=d2mat, d2mat0=d2mat0, name=name, duration=duration)


# functions


def matrix_setup(mat, mat0, axes=None):
    """ setup matrix operator """
    mat = matrix_format(mat)
    if mat0 is not None:
        mat0 = matrix_format(mat0)
        mat, mat0 = np.broadcast_arrays(mat, mat0)

    # axes
    if axes is not None:
        mat = oplinear.set_axes(2, mat, axes)
        mat0 = None if mat0 is None else oplinear.set_axes(2, mat0, axes)

    return mat, mat0

def matrix_format(mat, check=True):
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


def matrix_combine(mat1, mat2, mat01=None, mat02=None):
    """combine 2 matrix operators"""
    xp = common.get_array_module(mat1, mat2)
    mat1, mat2, mat01, mat02 = oplinear.extend_operators(2, mat1, mat2, mat01, mat02)
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

def matrix_apply(mat, mat0, sm):
    sm.states = matrix_prod(mat, sm.states, inplace=True)
    if mat0 is not None:
        sm.states += matrix_prod(mat0, sm.equilibrium, inplace=False)
    return sm

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
