"""array operations"""

import os
import logging
import warnings
import numpy as np

# get environment variable with LOG_LEVEL
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARN").upper())

LOGGER = logging.getLogger(__name__)

# cuda
DEFAULT_ARRAY_MODULE = "numpy"
_xp = None

#
# array module getters/setters


def set_array_module(xp=None):
    """set array module (eg. numpy, cupy)"""
    global _xp

    if not xp and _xp is not None:
        # already set
        return _xp

    elif not xp:
        # set default module
        xp = os.environ.get("ARRAY_MODULE", DEFAULT_ARRAY_MODULE)

    if xp in ["numpy", "np", np]:
        # setting global _xp to numpy
        _xp = np

    elif xp in ["cupy", "cp"] or getattr(xp, "__name__", None) == "cupy":
        try:
            # setting global _xp to cupy
            import cupy as cp
            import cupyx

            LOGGER.info("Setting array module to `cupy`")
            _xp = cp
        except ImportError:
            warnings.warn("cupy not found, falling back to numpy", RuntimeWarning)
            _xp = np

    # return global _xp
    return _xp


def get_array_module(*objs):
    """return module array"""
    global _xp
    if _xp is None:
        _xp = set_array_module()
    if objs:
        modules = _get_array_module(objs)
        if not modules:
            if all(map(isscalar, objs)):
                return np
            else:
                return _xp
        elif len(modules) > 1:
            raise ValueError(f"Found mixed array modules: {modules}")
        elif "numpy" in modules:
            return np
        elif "cupy" in modules:
            import cupy

            return cupy
    else:
        return _xp


def is_array_module(obj, xp="numpy"):
    """check array module"""
    if xp in ["numpy", "np", np]:
        xp = "numpy"
    elif xp in ["cupy", "cp"] or getattr(xp, "__name__", None) == "cupy":
        xp = "cupy"
    modules = _get_array_module(obj)
    if len(modules) == 1:
        return modules.pop() == xp
    else:
        return False


def _get_array_module(obj):
    """recursive getmodule for nested objects"""
    if type(obj).__name__ == "ndarray":
        return {type(obj).__module__.split(".")[0]}
    elif isinstance(obj, (list, tuple)):
        return set(item for items in map(_get_array_module, obj) for item in items)
    else:
        return set()


# array conversion


def asnumpy(arr, copy=False):
    """return numpy array"""
    if isinstance(arr, (list, tuple)):
        # list or tuple
        return type(arr)(asnumpy(item, copy=copy) for item in arr)
    elif is_array_module(arr, "cupy"):
        # cupy
        xp = get_array_module(arr)
        return xp.asarray(arr).get()
    # else return numpy
    if copy:
        return np.copy(arr)
    return np.asarray(arr)


def asarray(obj, **kwargs):
    """return xp asarray"""
    xp = get_array_module()
    return xp.asarray(obj, **kwargs)


def return_arrays(func):
    """decorator for converting outputs to xp.ndarray"""

    def wrapped(*args, **kwargs):
        xp = get_array_module()
        arrays = func(*args, **kwargs)
        if isinstance(arrays, tuple):
            return tuple(xp.asarray(arr) for arr in arrays)
        return xp.asarray(arrays)

    return wrapped


def map_arrays(arrays=None, func=np.asarray, *, xp=np, **kwargs):
    """comvert list/tuple/dict of objects into arrays"""

    def _apply(value):
        if isscalar(value):
            return value
        return func(value)

    arrays = kwargs if arrays is None else arrays
    if isinstance(arrays, (list, tuple)):
        return type(arrays)(map(_apply, arrays))

    elif isinstance(arrays, dict):
        return {key: _apply(arrays[key]) for key in arrays}

    # else: scalar
    return _apply(arrays)


class ArrayTuple(tuple):
    """summable tuples"""

    def __neg__(self):
        return ArrayTuple(None if a is None else -a for a in self)

    def __add__(self, other):
        if isscalar(other):
            return ArrayTuple(other if a is None else a + other for a in self)
        return ArrayTuple(
            a if b is None else b if a is None else a + b
            for a, b in zip(self, other, strict=True)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isscalar(other):
            return ArrayTuple(other if a is None else a.__iadd__(other) for a in self)
        return ArrayTuple(
            (
                a
                if b is None
                else b if a is None else a + b if isscalar(a) else a.__iadd__(b)
            )
            for a, b in zip(self, other, strict=True)
        )

    def __mul__(self, other):
        if isscalar(other):
            # return ArrayTuple(0 * other if a is None else a * other for a in self)
            return ArrayTuple(None if a is None else a * other for a in self)
        return ArrayTuple(
            (
                None
                if (a is None) or (b is None)
                else
                # 0 * a if b is None else
                # 0 * b if a is None else
                a * b
            )
            for a, b in zip(self, other, strict=True)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        if isscalar(other):
            # return ArrayTuple(other * 0 if a is None else a.__imul__(other) for a in self)
            return ArrayTuple(None if a is None else a.__imul__(other) for a in self)
        return ArrayTuple(
            (
                None
                if (a is None) or (b is None)
                else
                # 0 * a if b is None else
                # 0 * b if a is None else
                a * b if isscalar(a) else a.__imul__(b)
            )
            for a, b in zip(self, other, strict=True)
        )


#
# wrapper of frequently used np/xp functions


def expand_dims(arr, dims):
    xp = get_array_module(arr)
    dims = tuple(dims) if hasattr(dims, "__len__") else dims
    return xp.expand_dims(arr, dims)


def broadcast_to(arr, shape):
    xp = get_array_module(arr)
    return xp.broadcast_to(arr, shape)


def isscalar(value):
    """replace np.isscalar so that np.array(1) is scalar"""
    try:
        len(value) > 0
        return False
    except TypeError:
        return True


def atleast_nd(arr, ndim):
    """atleast ndim"""
    xp = get_array_module(arr)
    shape = get_shape(arr)
    diff = max(ndim - len(shape), 0)
    return xp.reshape(arr, shape + (1,) * diff)


#
# reshape/broadcast/expand


def get_shape(arr):
    """Return shape of nested sequence
    cf. https://leanpub.com/fizzbuzz
    """
    if hasattr(arr, "shape"):
        return arr.shape
    elif hasattr(arr, "__len__"):
        # More dimensions, so make a recursive call
        outermost_size = len(arr)
        row_shape = get_shape(arr[0])
        return (outermost_size, *row_shape)
    else:
        # No more dimensions, so we're done
        return ()


def expand_shapes(*shapes, append=False):
    """expand shapes to common dimension"""
    ndim = max([len(shape) for shape in shapes])
    if not append:
        # prepend new dimensions
        shapes = [(1,) * max(ndim - len(shape), 0) + tuple(shape) for shape in shapes]
    else:  # append new dimensions
        shapes = [tuple(shape) + (1,) * max(ndim - len(shape), 0) for shape in shapes]
    return shapes


def broadcastable(*shapes, append=False):
    """check shapes can be expanded and broadcasted"""
    shapes = expand_shapes(*shapes, append=append)
    return all(len(set(dims) - {1}) <= 1 for dims in zip(*shapes))


def broadcast_shapes(*shapes, append=False):
    """compute shape after broadcasting (append or prepend new dimensions)"""
    ndim = max([len(shape) for shape in shapes])
    shapes = expand_shapes(*shapes, append=append)

    common = [1] * ndim
    for i in range(ndim):
        dims = list({shape[i] for shape in shapes if shape[i] > 1})
        if not dims:
            continue
        elif len(dims) > 1:
            raise ValueError(f"Incompatible shapes: {shapes}")
        common[i] = dims[0]
    return tuple(common)


def expand_arrays(*objs, append=False):
    """prepend (or append) new axes to arrays such that they all have the same number of dimensions

    args:
        *objs: arrays to expand (scalars/objects are passed through)
    """
    if not objs:
        return objs
    xp = get_array_module(*objs)
    shapes = [get_shape(arr) for arr in objs]
    if not broadcastable(*shapes, append=append):
        raise ValueError("ArrayTuple cannot be broadcast to a single shape")
    ndim = max(len(shape) for shape in shapes)

    return tuple(
        (
            xp.expand_dims(
                xp.asarray(arr),
                (
                    tuple(range(ndim - len(shape)))
                    if not append
                    else tuple(range(len(shape), ndim))
                ),
            )
            if shape
            else arr
        )
        for arr, shape in zip(objs, shapes)
    )


def set_axes(ndim, arr, axes):
    """extend array given list of axes"""
    ndim = arr.ndim - ndim
    if isinstance(axes, int):
        axes = tuple(range(axes, axes + ndim))
    elif not isinstance(axes, tuple) or not all(isinstance(ax, int) for ax in axes):
        raise ValueError(f"Invalid axes: {axes}")

    # expand dimensions
    newdims = tuple([i for i in range(max(axes)) if not i in axes])
    return expand_dims(arr, newdims)


def extend_operators(ndim, *ops):
    """extend operators to make them broadcastable
    TODO: remove? (duplicate of extend_arrays?)
    """
    shapes = [get_shape(op)[:-ndim] for op in ops]
    shape = broadcast_shapes(*shapes, append=True)
    ndim = len(shape)
    extended = []
    for op in ops:
        if op is None:
            extended.append(None)
        else:
            dims = tuple(range(op.ndim - ndim, ndim))
            extended.append(expand_dims(op, dims))
    return extended


#
# representation


def repr_operator(cls, names, values, fmts=None):
    fmts = fmts or [""] * len(names)
    args = []
    for name, value, fmt in zip(names, values, fmts):
        if value is None:
            continue
        else:
            value = repr_value(value, fmt)
        if not name:
            args.append(value)
        else:
            args.append(f"{name}={value}")

    return f'{cls}({", ".join(args)})'


def repr_value(value, fmt):
    if isscalar(value):
        return f"{value:{fmt}}"
    else:
        shape = get_shape(value)
        return f'({"x".join(map(str, shape))})'


#


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
