""" Linear operators and functions """
import abc
from . import diff, common


class LinearOperator(diff.DiffOperator, abc.ABC):
    """Base class for linear operators"""

    @abc.abstractmethod
    def __matmul__(self, other):
        pass


# functions


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
