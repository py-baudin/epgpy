""" Scalar operator and functions """
import numpy as np
from . import common, operator, diff, opscalar

NAX = np.newaxis


class MatrixOp(diff.DiffOperator, operator.CombinableOperator):
    """state-wise matrix multiplication operator"""

    def __init__(self, mat, mat0=None, *, dmats=None, d2mats=None, axes=None, check=True, **kwargs):
        """Initialize operator

        Args:
            mat: sequence of 3x3 matrices to multiply with state matrix
            mat0: sequence of 3x3 matrices to multiply with equilibrium matrix
            axes: shift axes to given indices

        """
        # init parent class
        super().__init__(**kwargs)
        self._init(mat, mat0, dmats=dmats, d2mats=d2mats, axes=axes, check=check)

    def _init(self, mat, mat0=None, *, dmats=None, d2mats=None, axes=None, check=True):
        """ initialize arrays """
        # setup matrix operator
        self.mat, self.mat0 = matrix_setup(mat, mat0, axes=axes, check=check)

        # setup derivatives
        dmats = dmats or {}
        d2mats = d2mats or {}
        self.dmats = {param: matrix_setup(*dmats[param], axes=axes, check=check) for param in dmats}
        self.d2mats = {params: matrix_setup(*d2mats[params], axes=axes, check=check) for params in d2mats}


    @property
    def shape(self):
        return self.mat.shape[:-2]

    def _apply(self, sm):
        """Apply transform to state matrix (inplace)"""
        return matrix_apply(self.mat, self.mat0, sm)

    def _derive1(self, sm, param):
        dmat, dmat0 = self.dmats[param]
        return matrix_apply(dmat, dmat0, sm)

    def _derive2(self, sm, params):
        d2mat, d2mat0 = self.d2mats[params]
        return matrix_apply(d2mat, d2mat0, sm)
    
    # combine

    @classmethod
    def combinable(cls, other):
        return isinstance(other, (cls, opscalar.ScalarOp))
    
    @classmethod
    def _combine(cls, op1, op2, **kwargs):
        """ combine multiple scalar operators"""
        
        # merge parameters and coefficients
        parameters = set(op1.parameters) | set(op2.parameters)
        coeffs1 = {var: op.coeffs1[var] for op in (op1, op2) for var in op.coeffs1}
        coeffs2 = {vars: op.coeffs2[vars] for op in (op1, op2) for vars in op.coeffs2}

        # combine arrays
        mats = op1.mat, op1.mat0
        dmats = getattr(op1, 'dmats', {})
        d2mats = getattr(op1, 'd2mats', {})
            
        def apply(mats):
            return matrix_combine(mats[0], op2.mat, mats[1], None)
        
        def derive1(mats, param):
            dmat, dmat0 = op2.dmats[param]
            return matrix_combine(mats[0], dmat, mats[1], dmat0)
        
        def derive1_2(mats, param):
            dmat, _ = op2.dmats[param]
            return matrix_combine(mats[0], dmat, mats[1], None)
            
        def derive2(mats, params):
            d2mat, d2mat0 = op2.d2mats[params]
            return matrix_combine(mats[0], d2mat, mats[1], d2mat0)

        # combine operators
        if dmats or op2.coeffs1 or d2mats or op2.coeffs2:
            # combine differential operators
            d2mats = op2._apply_order2(mats, dmats, d2mats, apply=apply, derive1=derive1_2, derive2=derive2)
            dmats = op2._apply_order1(mats, dmats, apply=apply, derive1=derive1)

        mats = matrix_combine(mats[0], op2.mat, mats[1], op2.mat0)
        
        return MatrixOp(
            mats[0], mats[1], 
            dmats=dmats, d2mats=d2mats, 
            parameters=parameters,
            order1=coeffs1,
            order2=coeffs2,
            **kwargs,
        )



# functions


def matrix_setup(mat, mat0, axes=None, check=True):
    """ setup matrix operator """
    mat = matrix_format(mat, check=check)
    if mat0 is not None:
        mat0 = matrix_format(mat0, check=check)
        mat, mat0 = np.broadcast_arrays(mat, mat0)

    # axes
    if axes is not None:
        mat = common.set_axes(2, mat, axes)
        mat0 = None if mat0 is None else common.set_axes(2, mat0, axes)

    return common.ArrayTuple([mat, mat0])

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
    mat1, mat2, mat01, mat02 = common.extend_operators(2, mat1, mat2, mat01, mat02)
    mat = xp.einsum("...ij,...jk->...ik", mat2, mat1)
    if mat01 is None and mat02 is None:
        mat0 = None
    elif mat01 is None:
        mat0 = mat02.copy()
    else:
        mat0 = xp.einsum("...ij,...jk->...ik", mat2, mat01)
        if mat02 is not None:
            mat0 += mat02
    # return mat, mat0
    return common.ArrayTuple([mat, mat0])


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
