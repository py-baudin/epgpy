from collections import abc
import numpy as np


class ArrayCollection:
    """store collection of broadcastable arrays"""

    def __init__(self, *, kdim=0, xp=np):
        self._arrays = {}
        self._shape = ()
        self.kdim = kdim  # number of free dimensions
        self.xp = xp

    def __repr__(self):
        arrays = str(list(self._arrays))
        return f"ArrayCollection({self.shape}, {arrays})"

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def _update(self):
        """update common shape"""
        shapes = [arr.shape[: arr.ndim - self.kdim] for arr in self._arrays.values()]
        self._shape = np.broadcast_shapes(*shapes)

    def pop(self, name, *default):
        """pop array"""
        arr = self._arrays.pop(name, *default)
        self._update()
        return arr

    def set(self, name, arr):
        """set array"""
        xp = self.xp
        # always copy
        arr = xp.array(arr)
        # store array
        self._arrays[name] = arr
        self._update()

    def get(self, name, *default):
        """get array (or broadcast view of)"""
        xp = self.xp
        if not name in self._arrays and default:
            return default[0]
        arr = self._arrays[name]
        return self.broadcast(arr)

    def broadcast(self, arr):
        """broadcast array to match collection's shape"""
        xp = self.xp
        arr = xp.asarray(arr)
        shape = self._shape + arr.shape[arr.ndim - self.kdim :]
        if arr.shape != shape:
            arr = xp.broadcast_to(arr, shape)
        return arr

    def expand(self, ndim, insert_index=0):
        """add dimensions to all arrays"""
        xp = self.xp
        for name, arr in self._arrays.items():
            shape = arr.shape[: arr.ndim - self.kdim]
            start = tuple(range(len(shape)))[insert_index]
            dims = tuple(range(start, start + ndim - len(shape)))
            self._arrays[name] = xp.expand_dims(arr, dims)
        self._update()

    def reduce(self, ndim, remove_index=0):
        xp = self.xp
        for name, arr in self._arrays.items():
            shape = arr.shape[: arr.ndim - self.kdim]
            start = remove_index if remove_index >= 0 else len(shape) - ndim
            take = [
                0 if (i >= start) and (i < start + ndim) else slice(None)
                for i in range(arr.ndim)
            ]
            self._arrays[name] = arr[tuple(take)]
        self._update()

    def resize(self, shape):
        """broadcast and copy arrays"""
        xp = self.xp
        shape = tuple(shape)
        for name, arr in self._arrays.items():
            arr = self.get(name)
            _shape = shape + arr.shape[arr.ndim - self.kdim :]
            self._arrays[name] = xp.copy(xp.broadcast_to(arr, _shape))
        self._update()

    def resize_axis(self, size, axis):
        """resize array at given axis (pad with zeros if necessary)"""
        xp = self.xp
        for name, arr in self._arrays.items():
            ax = axis if axis >= 0 else arr.ndim - self.kdim + axis
            diff = size - arr.shape[ax]
            if diff > 0:
                pad = [
                    (diff // 2, (diff + 1) // 2) if i == ax else (0, 0)
                    for i in range(arr.ndim)
                ]
                self._arrays[name] = xp.pad(arr, pad)
            elif diff < 0:
                index = [
                    slice(-diff // 2, arr.shape[i] + diff // 2)
                    if i == ax
                    else slice(None)
                    for i in range(arr.ndim)
                ]
                self._arrays[name] = arr[tuple(index)]
        self._update()
