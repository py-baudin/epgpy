"""Scalar operator and functions"""

import abc
import numpy as np
from . import common, operator, diff

NAX = np.newaxis
SL = slice(None)


class ScalarOp(diff.DiffOperator, operator.CombinableOperator):
    """State-wise scalar multiplication operator"""

    def __init__(
        self,
        arr,
        arr0=None,
        *,
        darrs=None,
        d2arrs=None,
        axes=None,
        check=True,
        **kwargs,
    ):
        """Initialize operator"""

        super().__init__(
            parameters_order1=set(darrs or []),
            parameters_order2=set(d2arrs or []),
            **kwargs,
        )
        self._init(arr, arr0, darrs=darrs, d2arrs=d2arrs, axes=axes, check=check)

    def _init(self, arr, arr0=None, *, darrs=None, d2arrs=None, axes=None, check=True):
        """initialize arrays"""

        # setup scalar operator
        self.arr, self.arr0 = scalar_setup(arr, arr0, axes=axes, check=check)

        # setup derivatives
        darrs = darrs or {}
        d2arrs = d2arrs or {}
        opts = {"axes": axes, "check": check, "ref": self.arr}
        self.darrs = {param: scalar_setup(*darrs[param], **opts) for param in darrs}
        self.d2arrs = {
            diff.Pair(params): scalar_setup(*d2arrs[params], **opts)
            for params in d2arrs
        }

    @property
    def shape(self):
        return self.arr.shape[:-1]

    @property
    def mat(self):
        return as_matrix(self.arr)

    @property
    def mat0(self):
        return as_matrix(self.arr0)

    @property
    def dmats(self):
        return {
            var: common.ArrayTuple(map(as_matrix, arrs))
            for var, arrs in self.darrs.items()
        }

    @property
    def d2mats(self):
        return {
            vars: common.ArrayTuple(map(as_matrix, arrs))
            for vars, arrs in self.d2arrs.items()
        }

    def _apply(self, sm):
        """apply operator inplace"""
        return scalar_apply(self.arr, self.arr0, sm)

    def _derive1(self, sm, param):
        """apply 1st order differential operator inplace"""
        darr, darr0 = self.darrs[param]
        return scalar_apply(darr, darr0, sm)

    def _derive2(self, sm, params):
        """apply 2nd order differential operator inplace"""
        d2arr, d2arr0 = self.d2arrs[params]
        return scalar_apply(d2arr, d2arr0, sm)

    def combinable(self, other):
        return isinstance(other, type(self))

    def copy(self, **kwargs):
        new = super().copy(**kwargs)
        new.arr = self.arr.copy()
        new.arr0 = None if self.arr0 is None else self.arr0.copy()
        new.darrs = self.darrs.copy()
        new.d2arrs = self.d2arrs.copy()
        return new

    @classmethod
    def _combine(cls, op1, op2, **kwargs):
        """combine multiple scalar operators"""

        # merge parameters and coefficients
        order1 = {param: op.order1[param] for op in (op1, op2) for param in op.order1}
        order2 = {pair for op in (op1, op2) for pair in op.order2}

        # combine arrays
        arrs = op1.arr, op1.arr0
        darrs = getattr(op1, "darrs", {})
        d2arrs = getattr(op1, "d2arrs", {})

        def derive0(arrs, inplace=False):
            return scalar_combine(arrs[0], op2.arr, arrs[1], None)

        def derive1(arrs, param, inplace=False):
            darr, darr0 = op2.darrs[param]
            return scalar_combine(arrs[0], darr, arrs[1], darr0)

        def derive1_2(arrs, param, inplace=False):
            darr, _ = op2.darrs[param]
            return scalar_combine(arrs[0], darr, arrs[1], None)

        def derive2(arrs, params, inplace=False):
            d2arr, d2arr0 = op2.d2arrs[params]
            return scalar_combine(arrs[0], d2arr, arrs[1], d2arr0)

        # combine operators
        if d2arrs or op2.order2:
            d2arrs = op2._apply_order2(
                arrs, darrs, d2arrs, derive0=derive0, derive1=derive1_2, derive2=derive2
            )
        if darrs or op2.order1:
            darrs = op2._apply_order1(arrs, darrs, derive0=derive0, derive1=derive1)

        arrs = scalar_combine(arrs[0], op2.arr, arrs[1], op2.arr0)

        return ScalarOp(
            arrs[0],
            arrs[1],
            darrs=darrs,
            d2arrs=d2arrs,
            order1=order1,
            order2=order2,
            **kwargs,
        )


#
# functions


def as_matrix(arr):
    if arr is None:
        return None
    xp = common.get_array_module(arr)
    return arr[..., NAX] * xp.eye(3)


def scalar_setup(arr, arr0=None, *, axes=None, check=True, ref=None):
    """setup scalar operator"""
    xp = common.get_array_module()
    arr = scalar_format(arr, check=check)
    if ref is not None:
        arr, _ = xp.broadcast_arrays(arr, ref)

    if arr0 is not None:
        arr0 = scalar_format(arr0, check=check)
        arr, arr0 = xp.broadcast_arrays(arr, arr0)

    if axes is not None:
        arr = common.set_axes(1, arr, axes)
        arr0 = None if arr0 is None else common.set_axes(1, arr0, axes)
    return common.ArrayTuple([arr, arr0])


def scalar_format(arr, check=True):
    """setup array for scalar operator"""
    xp = common.get_array_module()
    arr = xp.asarray(arr)
    if arr.ndim == 1:
        arr = arr[NAX]

    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected ...x3 array shape, found: {arr.shape}")
    elif not check:
        return arr
    elif check and not xp.allclose(arr, arr[..., (1, 0, 2)].conj()):
        raise ValueError(f"Invalid coefficients: {arr}")

    return xp.asarray(arr)


def scalar_combine(arr_1, arr_2, arr0_1=None, arr0_2=None):
    """combine 2 scalar operators"""
    xp = common.get_array_module()
    arr_1, arr_2, arr0_1, arr0_2 = common.extend_operators(
        1, arr_1, arr_2, arr0_1, arr0_2
    )
    arr = arr_2 * arr_1
    if arr0_1 is None and arr0_2 is None:
        arr0 = None
    elif arr0_1 is None:
        arr0 = arr0_2.copy()
    else:
        arr0 = arr_2 * arr0_1
        if arr0_2 is not None:
            arr0 += arr0_2
    return common.ArrayTuple([arr, arr0])


def scalar_apply(arr, arr0, sm):
    states = sm.states
    states = scalar_prod(arr, states, inplace=True)
    if arr0 is not None:
        states += scalar_prod(arr0, sm.equilibrium)
    sm.states = states
    return sm


def scalar_prod(arr, states, inplace=False):
    """element-wise product product"""
    ndim = states.ndim - arr.ndim
    arr = arr[(...,) + (NAX,) * ndim + (SL,)] if ndim > 1 else arr[..., NAX, :]
    if inplace:
        try:
            states *= arr
            return states
        except ValueError:
            pass  # inplace not feasible
    return states * arr
