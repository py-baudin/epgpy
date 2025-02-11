"""Scalar operator and functions"""

import numpy as np
from . import common, operator, diff, opscalar

NAX = np.newaxis
SL = slice(None)


class MatrixOp(diff.DiffOperator, operator.CombinableOperator):
    """state-wise matrix multiplication operator"""

    def __init__(
        self,
        mat,
        mat0=None,
        *,
        dmats=None,
        d2mats=None,
        axes=None,
        check=True,
        **kwargs,
    ):
        """Initialize operator

        Args:
            mat: sequence of 3x3 matrices to multiply with state matrix
            mat0: sequence of 3x3 matrices to multiply with equilibrium matrix
            axes: shift axes to given indices

        """
        # init parent class
        super().__init__(
            parameters_order1=set(dmats or []),
            parameters_order2=set(d2mats or []),
            **kwargs,
        )
        self._init(mat, mat0, dmats=dmats, d2mats=d2mats, axes=axes, check=check)

    def _init(self, mat, mat0=None, *, dmats=None, d2mats=None, axes=None, check=True):
        """initialize arrays"""
        # setup matrix operator
        self.mat, self.mat0 = matrix_setup(mat, mat0, axes=axes, check=check)

        # setup derivatives
        dmats = dmats or {}
        d2mats = d2mats or {}
        self.dmats = {
            param: matrix_setup(*dmats[param], axes=axes, check=check)
            for param in dmats
        }
        self.d2mats = {
            diff.Pair(params): matrix_setup(*d2mats[params], axes=axes, check=check)
            for params in d2mats
        }

    @property
    def shape(self):
        return self.mat.shape[:-2]

    def _apply(self, sm):
        """apply operator inplace"""
        return matrix_apply(self.mat, self.mat0, sm)

    def _derive1(self, sm, param):
        """apply 1st order differential operator inplace"""
        dmat, dmat0 = self.dmats[param]
        return matrix_apply(dmat, dmat0, sm)

    def _derive2(self, sm, params):
        """apply 1st order differential operator inplace"""
        d2mat, d2mat0 = self.d2mats[params]
        return matrix_apply(d2mat, d2mat0, sm)

    def copy(self, **kwargs):
        new = super().copy(**kwargs)
        new.mat = self.mat.copy()
        new.mat0 = None if self.mat0 is None else self.mat0.copy()
        new.dmats = self.dmats.copy()
        new.d2mats = self.d2mats.copy()
        return new

    # combine

    @classmethod
    def combinable(cls, other):
        return isinstance(other, (cls, opscalar.ScalarOp))

    @classmethod
    def _combine(cls, op1, op2, **kwargs):
        """combine multiple scalar operators"""

        # merge parameters and coefficients
        order1 = {param: op.order1[param] for op in (op1, op2) for param in op.order1}
        order2 = {pair for op in (op1, op2) for pair in op.order2}

        # combine arrays
        mats = op1.mat, op1.mat0
        dmats = getattr(op1, "dmats", {})
        d2mats = getattr(op1, "d2mats", {})

        def derive0(mats, inplace=False):
            return matrix_combine(mats[0], op2.mat, mats[1], None)

        def derive1(mats, param, inplace=False):
            dmat, dmat0 = op2.dmats[param]
            return matrix_combine(mats[0], dmat, mats[1], dmat0)

        def derive1_2(mats, param, inplace=False):
            dmat, _ = op2.dmats[param]
            return matrix_combine(mats[0], dmat, mats[1], None)

        def derive2(mats, params, inplace=False):
            d2mat, d2mat0 = op2.d2mats[params]
            return matrix_combine(mats[0], d2mat, mats[1], d2mat0)

        # combine operators
        if d2mats or op2.order2:
            d2mats = op2._apply_order2(
                mats, dmats, d2mats, derive0=derive0, derive1=derive1_2, derive2=derive2
            )
        if dmats or op2.order1:
            dmats = op2._apply_order1(mats, dmats, derive0=derive0, derive1=derive1)

        mats = matrix_combine(mats[0], op2.mat, mats[1], op2.mat0)

        return MatrixOp(
            mats[0],
            mats[1],
            dmats=dmats,
            d2mats=d2mats,
            order1=order1,
            order2=order2,
            **kwargs,
        )


# functions


def matrix_setup(mat, mat0, axes=None, check=True):
    """setup matrix operator"""
    xp = common.get_array_module()
    mat = matrix_format(mat, check=check)
    if mat0 is not None:
        mat0 = matrix_format(mat0, check=check)
        mat, mat0 = xp.broadcast_arrays(mat, mat0)

    # axes
    if axes is not None:
        mat = common.set_axes(2, mat, axes)
        mat0 = None if mat0 is None else common.set_axes(2, mat0, axes)

    return common.ArrayTuple([mat, mat0])


def matrix_format(mat, check=True):
    """setup matrix operator"""
    xp = common.get_array_module()
    mat = xp.asarray(mat)
    if mat.ndim == 2:
        mat = mat[NAX]
    if mat.ndim < 3 or mat.shape[-2:] != (3, 3):
        raise ValueError(f"Expected ...x3x3 array shape, found: {mat.shape}")
    elif not check:
        return mat
    elif not xp.allclose(mat, mat[..., (1, 0, 2), :][..., (1, 0, 2)].conj()):
        raise ValueError(f"Invalid matrix coefficients: {mat}")

    return xp.asarray(mat)


def matrix_combine(mat1, mat2, mat01=None, mat02=None):
    """combine 2 matrix operators"""
    xp = common.get_array_module()
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
    xp = common.get_array_module()
    mat = mats[0]
    for mat_ in mats[1:]:
        mat = xp.einsum("...ij,...jk->...ik", mat_, mat)
    return mat


def matrix_apply(mat, mat0, sm):
    states = sm.states
    states = matrix_prod(mat, states, inplace=True)
    if mat0 is not None:
        states += matrix_prod(mat0, sm.equilibrium)
    sm.states = states
    return sm


def matrix_prod(mat, states, inplace=False):
    """matrix multiplication"""
    xp = common.get_array_module()
    ndim = states.ndim - mat.ndim + 1
    mat = mat[(...,) + (NAX,) * ndim + (SL, SL)] if ndim > 1 else mat[..., NAX, :, :]

    if inplace:
        try:
            return xp.matmul(
                mat, states, axes=[(-2, -1), (-1, -2), (-1, -2)], out=states
            )
        except ValueError:
            pass  # inplace not feasible
    return xp.matmul(mat, states[..., xp.newaxis])[..., 0]
