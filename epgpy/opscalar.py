""" Scalar operator and functions """
import abc
import numpy as np
from . import common, operator, diff

NAX = np.newaxis


class ScalarOp(diff.DiffOperator, operator.CombinableOperator):
    """State-wise scalar multiplication operator"""

    def __init__(self, arr, arr0=None, *, darrs=None, d2arrs=None, axes=None, check=True, **kwargs):
        """Initialize operator"""

        super().__init__(**kwargs)
        self._init(arr, arr0, darrs=darrs, d2arrs=d2arrs, axes=axes, check=check)

    def _init(self, arr, arr0=None, *, darrs=None, d2arrs=None, axes=None, check=True):
        """ initialize arrays """

        # setup scalar operator
        self.arr, self.arr0 = scalar_setup(arr, arr0, axes=axes, check=check)

        # setup derivatives
        darrs = darrs or {}
        d2arrs = d2arrs or {}
        self.darrs = {param: scalar_setup(*darrs[param], axes=axes, check=check) for param in darrs}
        self.d2arrs = {params: scalar_setup(*d2arrs[params], axes=axes, check=check) for params in d2arrs}

        
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
        return {var: common.ArrayTuple(map(as_matrix, arrs)) for var, arrs in self.darrs.items()}
    
    @property
    def d2mats(self):
        return {vars: common.ArrayTuple(map(as_matrix, arrs)) for vars, arrs in self.d2arrs.items()}    
    
    def _apply(self, sm):
        """apply inplace"""
        return scalar_apply(self.arr, self.arr0, sm)
    
    def _derive1(self, sm, param):
        darr, darr0 = self.darrs[param]
        return scalar_apply(darr, darr0, sm)
    
    def _derive2(self, sm, params):
        d2arr, d2arr0 = self.d2arrs[params]
        return scalar_apply(d2arr, d2arr0, sm)
    
    def combinable(self, other):
        return isinstance(other, type(self))
            
    @classmethod
    def _combine(cls, op1, op2, **kwargs):
        """ combine multiple scalar operators"""
        
        # merge parameters and coefficients
        parameters = set(op1.parameters) | set(op2.parameters)
        coeffs1 = {var: op.coeffs1[var] for op in (op1, op2) for var in op.coeffs1}
        coeffs2 = {vars: op.coeffs2[vars] for op in (op1, op2) for vars in op.coeffs2}

        # combine arrays
        arrs = op1.arr, op1.arr0
        darrs = getattr(op1, 'darrs', {})
        d2arrs = getattr(op1, 'd2arrs', {})
            
        def derive0(arrs):
            return scalar_combine(arrs[0], op2.arr, arrs[1], None)
        
        def derive1(arrs, param):
            darr, darr0 = op2.darrs[param]
            return scalar_combine(arrs[0], darr, arrs[1], darr0)
        
        def derive1_2(arrs, param):
            darr, _ = op2.darrs[param]
            return scalar_combine(arrs[0], darr, arrs[1], None)
            
        def derive2(arrs, params):
            d2arr, d2arr0 = op2.d2arrs[params]
            return scalar_combine(arrs[0], d2arr, arrs[1], d2arr0)

        # combine operators
        if darrs or op2.coeffs1 or d2arrs or op2.coeffs2:
            # combine differential operators
            d2arrs = op2._apply_order2(arrs, darrs, d2arrs, derive0=derive0, derive1=derive1_2, derive2=derive2)
            darrs = op2._apply_order1(arrs, darrs, derive0=derive0, derive1=derive1)

        arrs = scalar_combine(arrs[0], op2.arr, arrs[1], op2.arr0)

        return ScalarOp(
            arrs[0], arrs[1], 
            darrs=darrs, d2arrs=d2arrs, 
            parameters=parameters,
            order1=coeffs1,
            order2=coeffs2,
            **kwargs,
        )


#
# functions

def as_matrix(arr):
    if arr is None:
        return None
    xp = common.get_array_module(arr)
    return arr[..., NAX] * xp.eye(3)


def scalar_setup(arr, arr0=None, *, axes=None, check=True):
    """ setup scalar operator """
    arr = scalar_format(arr, check=check)
    if arr0 is not None:
        arr0 = scalar_format(arr0, check=check)
        arr, arr0 = np.broadcast_arrays(arr, arr0)

    if axes is not None:
        arr = common.set_axes(1, arr, axes)
        arr0 = None if arr0 is None else common.set_axes(1, arr0, axes)
    return common.ArrayTuple([arr, arr0])


def scalar_format(arr, check=True):
    """setup array for scalar operator """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[NAX]

    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"Expected ...x3 array shape, found: {arr.shape}")
    elif not check:
        return arr
    elif check and not np.allclose(arr, arr[..., (1, 0, 2)].conj()):
        raise ValueError(f"Invalid coefficients: {arr}")

    xp = common.get_array_module()
    return xp.asarray(arr)


def scalar_combine(arr_1, arr_2, arr0_1=None, arr0_2=None):
    """combine 2 scalar operators"""
    xp = common.get_array_module(arr_1, arr_2)
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
    sm.states = scalar_prod(arr, sm.states, inplace=True)
    if arr0 is not None:
        sm.states += scalar_prod(arr0, sm.equilibrium, inplace=False)
    return sm


def scalar_prod(arr, states, *, inplace=False):
    """element-wise product product"""
    xp = common.get_array_module(arr, states)

    # expand mat dims if needed
    dims = tuple(range(arr.ndim - 1, states.ndim - 1))
    arr = xp.expand_dims(arr, dims)

    # broadcastable
    broadcastable = all(s1 <= s2 for s1, s2 in zip(arr.shape[:-1], states.shape[:-2]))

    # multiply
    if inplace and broadcastable:
        states *= arr
    else:
        states = states * arr
    return states


