import logging

import math
from . import common, utils, common

LOGGER = logging.getLogger(__name__)


class StateMatrix:
    """store the phase states of a n-dimensional system"""

    def __init__(
        self,
        init=None,
        *,
        density=1,
        equilibrium=None,
        coords=None,
        kvalue=1.0,
        tvalue=1.0,
        nstate=None,
        shape=None,
        check=True,
        **options,
    ):
        """Create n-dimensional phase-state matrix

        Parameters
        ===
            init: initial phase state coefficients
            equilibrium: state matrix at equilibrium
            coords: spatial/temporal coordinates of the phase states
            check: check `init` shape and resize matrix accordingly
            options:
                max_nstate: maximum number of phase-states
                kgrid: grid size for shift operator
        """
        self.arrays = ArrayCollection(expand_axis=-1)

        if equilibrium is None:
            # use density
            equilibrium = _init_states(density)

        # format equilibrium states
        equilibrium = _format_states(equilibrium, check=check)

        if init is None:
            # start from equilibrium
            init = equilibrium
        else:
            # custom init
            init = _format_states(init, check=check)

        # set initial magnetization
        self.arrays.set("states", init, layout=[..., "nstate", 3])

        # set equilibrium
        self.arrays.set(
            "equilibrium", equilibrium, layout=[..., "nstate", 3], resize=True
        )

        # set frequency coordinates
        if coords is not None:
            self.arrays.set("coords", coords, layout=[..., "nstate", "kdim"])
        self.kvalue = kvalue
        self.tvalue = tvalue

        if nstate:
            self.arrays.resize("nstate", 2 * nstate + 1)
        if shape:
            # broadcast and copy states
            self.arrays.broadcast(shape)
            self.arrays.set("states", self.arrays.get("states"))

        # additional metadata (eg. kgrid, max_nstate)
        self.options = options

        # add linked "system" collection
        self.system = ArrayCollection(expand_axis=-1)
        self.arrays.link(self.system)

    # public attributes

    @property
    def states(self):
        return self.arrays.get("states")

    @states.setter
    def states(self, value):
        # self.arrays.set("states", value, check=False)
        self.arrays.update("states", value)

    @property
    def density(self):
        return self.equilibrium[..., self.nstate, 2].real

    @property
    def equilibrium(self):
        """equilibrium state matrix with compatible shape/nstate"""
        return self.arrays.get("equilibrium")

    @property
    def coords(self):
        return self.arrays.get("coords", None, broadcast=False)

    @coords.setter
    def coords(self, value):
        self.arrays.set("coords", value)

    @property
    def ndim(self):
        """return number of dimensions of phase array"""
        return len(self.shape)

    @property
    def shape(self):
        """return shape of phase array"""
        return self.arrays.shape

    @property
    def size(self):
        """return size of phase array"""
        return math.prod(self.shape)

    @property
    def nstate(self):
        """return number of phase states"""
        return (self.arrays.axes["nstate"] - 1) // 2

    @property
    def kdim(self):
        """number of coords dimensions"""
        return 1 if self.coords is None else self.arrays.axes["kdim"]

    @property
    def i0(self):
        """index/indices of F0 state(s)"""
        if self.kdim < 4:
            return self.nstate
        xp = common.get_array_module()
        return xp.all(xp.isclose(self.coords[..., :3], 0), axis=-1)

    @property
    def F(self):
        """transversal states"""
        return self.states[..., 0]

    @property
    def F0(self):
        if self.kdim < 4:
            return self.states[..., self.nstate, 0]
        # select states with k==0
        return self.states[..., 0] * self.i0

    @property
    def Z(self):
        """longitudinal states"""
        return self.states[..., 2]

    @property
    def Z0(self):
        if self.kdim < 4:
            return self.states[..., self.nstate, 2]
        # select states with k==0
        return self.states[..., 2] * self.i0

    @property
    def k(self):
        """wavenumbers (coords first 3 dimensions)"""
        coords = self.coords
        if coords is None:
            coords = _setup_coords(self.nstate, 1, self.ndim)
        kvalue = self.kvalue
        if not common.isscalar(kvalue):
            kvalue = kvalue[: coords.shape[-1]]
        return coords[..., :3] * kvalue

    @property
    def t(self):
        """time-accumulated dephasing (coords 4th dimension)"""
        if self.kdim < 4:
            return 0
        return self.coords[..., 3] * self.tvalue

    @property
    def t0(self):
        """0-th state / time-accumulated dephasing (4th dimension)"""
        if self.kdim < 4:
            return 0
        return self.coords[..., 3] * self.i0 * self.tvalue

    @property
    def ktvalue(self):
        """concatenation of kvalue and tvalue (wavenumbers, accumulated time)"""
        kdim = self.kdim
        kvalue, tvalue = self.kvalue, self.tvalue
        if common.isscalar(kvalue):
            coeff = [kvalue] * min(kdim, 3) + [tvalue] * (kdim == 4)
        else:
            coeff = list(kvalue)[:3] + [tvalue] * (kdim == 4)
        return common.asarray(coeff)

    @property
    def norm(self):
        """state matrix norm"""
        return self.arrays.apply("states", utils.get_norm)

    @property
    def zeros(self):
        """zero state matrix with current shape/nstate"""
        return self.copy([[0, 0, 0]], nstate=self.nstate, shape=self.shape, check=False)

    @property
    def writeable(self):
        return True
        # return getattr(self.states.flags, "writeable", False)

    @property
    def array_module(self):
        return self.arrays.xp

    # operator overloading

    def __repr__(self):
        return f"StateMatrix({self.shape}, nstate={self.nstate})"

    def __array__(self):
        return common.asnumpy(self.states)

    def _cmp(self, other):
        """compare to another state matrix"""
        xp = common.get_array_module()
        if isinstance(other, StateMatrix):
            value = xp.asarray(other.states)
        elif xp.isscalar(other):
            value = other
        else:  # array
            value = xp.asarray(other)[..., xp.newaxis, xp.newaxis]
        return value

    def __add__(self, other):
        return self.copy(self.states + self._cmp(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self.states[:] += self._cmp(other)
        return self

    def __mul__(self, other):
        return self.copy(self.states * self._cmp(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self.states[:] *= self._cmp(other)
        return self

    def __eq__(self, other):
        return self.states == self._cmp(other)

    # public functions

    def copy(self, states=None, **kwargs):
        """copy state matrix"""
        sm = self.__new__(type(self))
        coll = self.arrays.copy()
        if states is not None:
            coll.update("states", states, resize=True)
        if "equilibrium" in kwargs:
            coll.update("equilibrium", kwargs.pop("equilibrium"), resize=True)
        if "coords" in kwargs:
            coll.update("coords", kwargs.pop("coords"), resize=True)
        sm.arrays = coll
        sm.kvalue = kwargs.pop("kvalue", self.kvalue)
        sm.tvalue = kwargs.pop("tvalue", self.tvalue)
        sm.options = {**self.options, **kwargs}
        sm.system = self.system
        return sm

    def resize(self, nstate):
        """resize state matrix to nstate"""
        if nstate == self.nstate:
            return
        self.arrays.resize("nstate", 2 * nstate + 1)

    def expand(self, ndim):
        """expend state matrix to n-dimensions"""
        diff = ndim - self.arrays.ndim
        if diff > 0:
            self.arrays.expand(diff)

    def reduce(self, ndim):
        """expend state matrix to n-dimensions"""
        diff = self.arrays.ndim - ndim
        if diff > 0:
            self.arrays.reduce(diff)

    def check(self):
        return utils.check_states(self.states)

    def setup_coords(self, kdim):
        if self.coords is not None:
            # resize
            diff = kdim - self.kdim
            if diff > 0:
                xp = self.arrays.xp
                coords = self.coords
                zeros = xp.zeros(coords.shape[:-1] + (diff,))
                coords = xp.concatenate([coords, zeros], axis=-1)
            elif diff < 0:
                raise RuntimeError("Cannot remove existimg k-dimension")
            else:
                return
        else:
            coords = _setup_coords(self.nstate, kdim)
        self.arrays.set("coords", coords, layout=[..., "nstate", "kdim"])

    def stack(self, seq, *, axis=0):
        """stack state matrices"""
        xp = common.get_array_module()
        seq = [self] + list(seq)
        states = xp.stack([sm.states for sm in seq], axis=axis)
        equibm = xp.stack([sm.equilibrium for sm in seq], axis=axis)
        coords = None
        if seq[0].coords is not None:
            coords = xp.stack([sm.coords for sm in seq], axis=axis)
        kwargs = {
            "kvalue": seq[0].kvalue,
            "tvalue": seq[0].tvalue,
            **seq[0].options,
        }
        cls = type(self)
        return cls(states, equilibrium=equibm, coords=coords, check=False, **kwargs)

    def unstack(self, sm=None, *, axis=0):
        """split state matrices at given axis"""
        xp = common.get_array_module()
        sm = self if sm is None else sm
        states, equibm, coords = sm.states, sm.equilibrium, sm.coords
        if axis != 0:
            states = xp.moveaxis(states, axis, 0)
            equibm = xp.moveaxis(equibm, axis, 0)
            coords = xp.moveaxis(coords, axis, 0) if coords is not None else None
        coords = [None] * len(states) if coords is None else coords
        kwargs = {
            "kvalue": sm.kvalue,
            "tvalue": sm.tvalue,
            **sm.options,
        }
        cls = type(self)
        return (
            cls(st, equilibrium=eq, coords=coo, check=False, **kwargs)
            for st, eq, coo in zip(states, equibm, coords)
        )

    def _stack(self, seq, *, axis=0):
        return type(self).stack([self] + list(seq), axis=axis)

    def _unstack(self, *, axis=0):
        return type(self).unstack(self, axis=axis)


# private functions


def _init_states(density=1):
    xp = common.get_array_module()
    density = xp.atleast_1d(density)
    density = xp.expand_dims(density, (density.ndim, density.ndim + 1))
    init = xp.array([[[0, 0, 1]]])
    states = density * init
    return states


def _format_states(states, check=True):
    """format and check states matrix"""
    # init state matrix
    xp = common.get_array_module()
    states = xp.asarray(states).astype(xp.complex128)

    if states.ndim == 1:
        # states is a 0-state vector
        if check and states.size != 3:
            raise ValueError("The number of state dimensions must be 3")
        states = states.reshape((1, 1, 3))

    elif states.ndim == 2:
        # states is a single-dimensional k-state matrix
        if check and states.shape[1] != 3:
            raise ValueError("The number of state dimensions must be 3")
        elif check and states.shape[0] % 2 != 1:
            raise ValueError("The number of states must be odd")
        k = (states.shape[0] - 1) // 2
        states = states.reshape((1, 2 * k + 1, 3))

    else:
        # states is a multi-dimensional k-state matrix
        if check and states.shape[-1] != 3:
            raise ValueError("The number of state dimensions must be 3")
        elif check and states.shape[-2] % 2 != 1:
            raise ValueError("The number of states must be odd")

    # check values
    if check:
        if not xp.allclose(states[..., 1], states[..., ::-1, 0].conj()):
            raise ValueError("The F-state columns do no match.")
        if not xp.allclose(states[..., 2], states[..., ::-1, 2].conj()):
            raise ValueError("The Z-state columns is not symmetrical.")
    return states


def _setup_coords(nstate, kdim=1, ndim=1):
    """setup wavenumbers array"""
    xp = common.get_array_module()
    coords = xp.c_[
        xp.arange(-nstate, nstate + 1), xp.zeros((2 * nstate + 1, kdim - 1), dtype=int)
    ]
    coords = common.expand_dims(coords, tuple(range(ndim)))
    return coords


#
# array collection


class ArrayCollection:
    """Broadcastable arrays

    Each array has a layout whose items define the broadcastable structure:
    Items:
        Ellipsis/...: broadcastable axes
        <int>: fixed axis
        <str>: named axis
        None: free axis

    Example:
        # whole array is broadcast
        coll.set('arr1', arr1, [...])

        # last axis of arr2 is free
        coll.set('arr2', arr2, [..., None])

        # first axis of arr3 is fixed with size==3
        coll.set('arr3', arr3, [3, ...])

        # first axis of arr4 and last axis of arr4 have the same size
        coll.set('arr4', arr4, ['ax', ...])
        coll.set('arr5', arr5, [..., 'ax'])

    """

    def __init__(self, default=None, *, expand_axis=0):
        self._shape = None
        self._expand_axis = int(expand_axis)
        self._layouts = {}
        self._arrays = {}
        self._shapes = {}
        self._axes = {}
        self._default = (1,) if default is None else tuple(default)
        self._linked = set()
        self._update_shape()

    def __len__(self):
        return len(self._arrays)

    def __repr__(self):
        return f"ArrayCollection({len(self._arrays)}, shape={self.shape})"

    def __iter__(self):
        return iter(self._arrays)

    def __contains__(self, name):
        return name in self._arrays

    def __getitem__(self, key):
        retval = self.get(key, default=...)
        if retval is Ellipsis:
            raise KeyError(key)
        return retval

    @property
    def xp(self):
        return common.get_array_module()

    @property
    def ndim(self):
        """broadcast ndim"""
        return len(self._shape)

    @property
    def shape(self):
        """return broadcast shape"""
        return self._shape

    @property
    def expand_axis(self):
        ax = self._expand_axis
        return ax if ax >= 0 else self.ndim + ax + 1

    @property
    def axes(self):
        """dictionary of named axes"""
        return self._axes

    def get(self, name, default=None, *, broadcast=True):
        """get (broadcast) array from collection"""
        if not name in self._arrays:
            return default
        array = self._arrays[name]
        if array.shape == self._shapes[name]:
            # skip broadcasting
            return array
        layout = self._layouts[name]
        return self._expand_and_broadcast(array, layout, broadcast=broadcast)

    def update(self, name, array, *, resize=False):
        """update array inplace"""
        try:
            self._arrays[name][:] = array
        except ValueError:
            self.set(name, array, check=False, resize=resize)

    def apply(self, name, func):
        """apply function to raw array"""
        return func(self._arrays[name])

    def set(self, name, array, *, layout=None, resize=False, check=True):
        """add array to collection"""
        xp = self.xp
        array = xp.array(array)  # copy array

        if name in self._arrays:
            if layout is None:  # keep existing layout
                layout = self._layouts[name]
        elif layout is None:  # set default layout
            layout = [Ellipsis]

        self.check_layout(layout)
        if resize:
            # resize named axes to match current shape
            axes = self.get_named_axes(ignore=name)
            for idx, ax in self._get_named_axes(array.ndim, layout):
                size = array.shape[idx]
                if ax in axes and size != axes[ax]:
                    array = self.resize_array(array, axes[ax] - size, axis=idx)
        if check:
            self.check_shape(array.shape, layout, ignore=name)
        self._arrays[name] = array
        self._layouts[name] = tuple(layout)
        self._update_shape()
        self._axes = self.get_named_axes()

    def pop(self, name, default=None):
        """remove array from collection"""
        if not name in self._arrays:
            return default
        self._layouts.pop(name)
        array = self._arrays.pop(name)
        self._update_shape()
        return array

    # utilities
    def copy(self):
        """copy collection"""
        coll = self.__new__(type(self))
        layouts, arrays = self._layouts, self._arrays
        # coll.xp = self.xp
        coll._expand_axis = self._expand_axis
        coll._shape = self._shape
        coll._arrays = {name: coll.xp.array(arrays[name]) for name in arrays}
        coll._layouts = dict(layouts)
        coll._shapes = dict(self._shapes)
        coll._axes = dict(self._axes)
        coll._default = self._default
        coll._linked = self._linked
        return coll

    def get_named_axes(self, *, ignore=None):
        """get named axes dimensions"""
        arrays, layouts = self._arrays, self._layouts
        return {
            ax: arr.shape[idx]
            for name, arr in arrays.items()
            if name != ignore
            for idx, ax in self._get_named_axes(arr.ndim, layouts[name])
        }

    def check_layout(self, layout):
        """check layout"""
        counts = [i for i, ax in enumerate(layout) if ax is Ellipsis]
        if not counts:
            raise ValueError(f"`layout` must contain one Ellipsis: {layout}")
        elif len(counts) > 1:
            raise ValueError(f"`layout` must contain only one Ellipsis: {layout}")

    def check_shape(self, shape, layout, *, ignore=None):
        """check shape"""
        shape = tuple(shape)
        axes = self.get_named_axes(ignore=ignore)

        # check named axes
        idx = 0
        for i, ax in enumerate(layout):
            if ax is Ellipsis:
                idx += len(shape) - len(layout) + 1
            elif isinstance(ax, int) and shape[idx] != ax:
                raise ValueError(f"Invalid axis dimension {idx} in array: {shape}")
            elif isinstance(ax, str) and shape[idx] != axes.get(ax, shape[idx]):
                raise ValueError(
                    f"Invalid axis dimension {idx} (`{ax}`) in array: {shape}"
                )
            else:
                idx += 1

        # check shape
        axis = layout.index(Ellipsis)
        common = list(shape[axis : len(shape) - len(layout) + 1])
        shared = self.shape
        diff = len(shared) - len(common)

        ax = self.expand_axis
        if diff < 0:
            common[ax : ax - diff] = []
        elif diff > 0:
            common = (
                common[:ax]
                + [
                    1,
                ]
                * diff
                + common[ax:]
            )
        if any(1 != d1 != d2 != 1 for d1, d2 in zip(common, shared)):
            raise ValueError(f"Incompatible shape: {shape} and {shared}")

    def check(self, shape, layout, *, ignore=None):
        """check shape and layout"""
        self.check_layout(layout)
        self.check_shape(shape, layout, ignore=ignore)

    def resize(self, ax, size, *, constant=0):
        """resize named axis"""
        # xp = self.xp
        axes = self.axes
        if not ax in axes:
            raise ValueError(f"Unknown size: {ax}")
        diff = size - axes[ax]
        if diff == 0:
            return
        for name in self._arrays:
            layout = self._layouts[name]
            if not ax in layout:
                continue
            arr = self._arrays[name]
            axis = layout.index(ax)
            if 0 <= layout.index(Ellipsis) < axis:
                axis = arr.ndim - len(layout) + axis
            arr = self.resize_array(arr, diff, axis=axis, constant=constant)
            self._arrays[name] = arr
            self._shapes[name] = self._get_broadcast_shape(
                self._shape, arr.shape, layout
            )
        self._axes = self.get_named_axes()

    def expand(self, ndim):
        """add dimensions to broadcast shape"""
        shape = list(self.shape)
        axis = self.expand_axis
        shape[axis:axis] = [1] * ndim
        self._default = tuple(shape)
        self._update_shape()

    def broadcast(self, shape):
        """broadcast collection to shape"""
        self.check(shape, [...])
        self._default = shape
        self._update_shape()

    def reduce(self, ndim, *, axis=0):
        """remove dimensions from default shape"""
        shape = list(self.shape)
        axis = self._expand_axis
        if axis >= 0:
            shape[axis : axis + ndim] = []
        else:
            axis = len(shape) + axis + 1
            shape[axis - ndim : axis + 1] = []
        self._default = tuple(shape)
        self._update_shape()

    def link(self, other):
        if not isinstance(other, ArrayCollection):
            raise ValueError(f"Not a collection: {other}")
        self._linked.add(other)
        self._update_shape()

    # private

    @staticmethod
    def _get_shared_axes(shape, layout):
        """return part of shape that will be broadcast"""
        start = layout.index(Ellipsis)
        end = len(shape) - (len(layout) - start - 1)
        return shape[start:end]

    @staticmethod
    def _get_shared_shape(ndim, shape, axis):
        """return shape after broadcasting"""
        if axis < 0:
            axis = ndim - axis + 1
        return shape[:axis] + (1,) * max(ndim - len(shape), 0) + shape[axis:]

    @staticmethod
    def _get_named_axes(ndim, layout):
        """return axes names and index"""
        idx = layout.index(Ellipsis)
        diff = ndim - len(layout)
        return [
            (i + (i > idx) * diff, ax)
            for i, ax in enumerate(layout)
            if i != idx and isinstance(ax, str)
        ]

    @staticmethod
    def _get_broadcast_shape(shared, shape, layout):
        """return shape after broadcasting"""
        shape = list(shape)
        start = layout.index(Ellipsis)
        end = len(shape) - (len(layout) - start - 1)
        shape[start:end] = shared
        return tuple(shape)

    def _update_shape(self):
        arrays, layouts = self._arrays, self._layouts
        shared = [
            self._get_shared_axes(arrays[name].shape, layouts[name]) for name in arrays
        ] + [self._default]
        ndim = max(len(shape) for shape in shared)
        shapes = [
            self._get_shared_shape(ndim, shape, self._expand_axis) for shape in shared
        ]
        self._shape = tuple(max(shape[i] for shape in shapes) for i in range(ndim))
        self._shapes = {
            name: self._get_broadcast_shape(
                self._shape, arrays[name].shape, layouts[name]
            )
            for name in arrays
        }
        for other in self._linked:
            other._default = self._shape
            other._update_shape()

    def _expand_and_broadcast(self, array, layout, broadcast=True):
        """expand and broadcast array to shared shape"""
        xp = self.xp

        # common shape
        shared = self.shape

        # expand dimensions
        diff = len(shared) - (array.ndim - len(layout) + 1)
        axis = layout.index(Ellipsis)
        if self._expand_axis < 0:
            insert = axis + self._expand_axis + array.ndim - len(layout) + 2
        else:
            insert = axis + self._expand_axis
        dims = tuple(insert + i for i in range(diff))

        # broadcast shape
        shape = list(array.shape)
        shape[insert:insert] = [1] * diff
        shape[axis : axis + len(shared)] = shared

        if dims:
            array = xp.expand_dims(array, dims)
        if broadcast and (tuple(shape) != array.shape):
            array = xp.broadcast_to(array, shape)
        return array

    def resize_array(self, array, diff, axis=0, *, constant=0):
        """resize array at given axis"""
        xp = self.xp
        if diff < 0:  # crop
            slices = [slice(None) for _ in range(array.ndim)]
            slices[axis] = slice(-diff // 2, array.shape[axis] - (-diff + 1) // 2)
            array = array[tuple(slices)]
        elif diff > 0:  # padd
            pad = [(0, 0) for _ in range(array.ndim)]
            pad[axis] = (diff // 2, (diff + 1) // 2)
            array = xp.pad(array, pad, constant_values=constant)
        return array
