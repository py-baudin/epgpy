""" Scalar operator and functions """
import abc
import numpy as np
from . import common, oplinear

NAX = np.newaxis


class ScalarOp(oplinear.LinearOperator):
    """State-wise scalar multiplication operator"""


    def __init__(self, arr, arr0=None, *, axes=None, check=True, **kwargs):
        """Initialize operator"""
        self.arr, self.arr0 = scalar_setup(arr, arr0, axes=axes, check=check)
        darrs, d2arrs = kwargs.pop('darrs', {}), kwargs.pop('d2arrs', {})
        self.darrs = {param: scalar_setup(*darrs[param], axes=axes, check=check) for param in darrs}
        self.d2arrs = {params: scalar_setup(*d2arrs[params], axes=axes, check=check) for params in d2arrs}
        super().__init__(**kwargs)
        
    @property
    def shape(self):
        return self.arr.shape[:-1]
    
    @property
    def mat(self):
        return as_matrix(self.arr)
    
    @property
    def mat0(self):
        return as_matrix(self.arr0)    
    
    def __matmul__(self, other):
        """multiply operators 
        
        
        put in oplinear ?"""
        if not isinstance(other, ScalarOp):
            return NotImplemented
        return self.combine([self, other])

    def _apply(self, sm):
        """apply inplace"""
        return scalar_apply(self.arr, self.arr0, sm)
    
    def _derive1(self, sm, param):
        darr, darr0 = self.darrs[param]
        return scalar_apply(darr, darr0, sm)
    
    def _derive2(self, sm, params):
        d2arr, d2arr0 = self.d2arrs[params]
        return scalar_apply(d2arr, d2arr0, sm)
    
    @staticmethod
    def combine(ops, *, name=None, duration=None, check=True):
        """ combine multiple scalar operators"""
        arrs = ops[0].arr, ops[0].arr0
        darrs = getattr(ops[0], 'darrs', {})
        d2arrs = getattr(ops[0], 'd2arrs', {})

        for op in ops[1:]:
            if not isinstance(op, ScalarOp):
                raise NotImplementedError()
            
            def apply(offset=True):
                def func(arrs, op=op):
                    return scalar_combine(arrs[0], op.arr, arrs[1], op.arr0 if offset else None)
                return func
            
            def derive1(offset=True):
                def func(arrs, param, op=op):
                    darr, darr0 = op.darrs[param]
                    return scalar_combine(arrs[0], darr, arrs[1], darr0 if offset else None)
                return func
                
            def derive2(arrs, params, op=op):
                d2arr, d2arr0 = op.d2arrs[params]
                return scalar_combine(arrs[0], d2arr, arrs[1], d2arr0)

            if darrs or op.coeffs1 or d2arrs or op.coeffs2:
                # combine differential operators
                d2arrs = op._apply_order2(arrs, darrs, d2arrs, apply=apply(False), derive1=derive1(False), derive2=derive2)
                darrs = op._apply_order1(arrs, darrs, apply=apply(False), derive1=derive1())

            # combine operators
            arrs = apply()(arrs)
            # arrs = scalar_combine(arrs[0], op.arr, arrs[1], op.arr0)

        if name is None:
            name = "|".join(op.name for op in ops)
        if duration is None:
            duration = sum(op.duration for op in ops)

        # for var in darrs:
        #     darrs[var] *= (1, None)
        for vars in d2arrs:
            d2arrs[vars] *= (1, None)
        
        parameters = {param for op in ops for param in op.parameters}
        coeffs1 = {var: op.coeffs1[var] for op in ops for var in op.coeffs1}
        coeffs2 = {vars: op.coeffs2[vars] for op in ops for vars in op.coeffs2}
        return ScalarOp(
            arrs[0], arrs[1], 
            darrs=darrs, d2arrs=d2arrs, 
            order1=coeffs1,
            order2=coeffs2,
            parameters=parameters,
            name=name, duration=duration,
            check=check,
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
        arr = oplinear.set_axes(1, arr, axes)
        arr0 = None if arr0 is None else oplinear.set_axes(1, arr0, axes)
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
    arr_1, arr_2, arr0_1, arr0_2 = oplinear.extend_operators(
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

