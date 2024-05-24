import logging

import numpy as np
from . import common, utils, common  # arraycollection

LOGGER = logging.getLogger(__name__)

""" todo

- remove arraycollection
 - replace with functions to maintain ndims for:
    - states (n1 x...x nstate x ndim)
    - kindices (n1 x...x nstate x kdim)
    - density (n1 x...x ndim)
    - init (1x... x nstate x 3)
- equilibrium is computed as (density * init)

rename: 
    `equilibrium` -> `eq` ?

"""


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
        # self._collection = arraycollection.ArrayCollection(kdim=1)
        self._collection = ArrayCollection(expand_axis=-1)

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

        # self._collection.set("states", init)
        self._collection.set("states", init, layout=[..., "nstate", 3])

        # self._collection.set("equilibrium", equilibrium)
        self._collection.set("equilibrium", equilibrium, layout=[..., 1, 3])
        if coords is not None:
            # self._collection.set("coords", coords)
            self._collection.set("coords", coords, layout=[..., "nstate", "kdim"])
        self.kvalue = kvalue
        self.tvalue = tvalue

        if nstate:
            # self._collection.resize_axis(2 * nstate + 1, -1)
            self._collection.resize("nstate", 2 * nstate + 1)
        if shape:
            # self._collection.resize(tuple(shape) + (2 * self.nstate + 1,))
            self._collection.broadcast(shape)

        # additional metadata (eg. kgrid, max_nstate)
        self.options = options

    # public attributes

    @property
    def states(self):
        return self._collection.get("states")

    @states.setter
    def states(self, value):
        # self._collection.set("states", value)
        self._collection.set("states", value)

    @property
    def density(self):
        # return self.equilibrium[..., self.nstate, 2].real
        return self.equilibrium[..., 0, 2].real

    @property
    def equilibrium(self):
        """equilibrium state matrix with compatible shape/nstate"""
        return self._collection.get("equilibrium")

    @property
    def coords(self):
        return self._collection.get("coords", None)

    @coords.setter
    def coords(self, value):
        self._collection.set("coords", value)

    @property
    def ndim(self):
        """return number of dimensions of phase array"""
        # return self.states.ndim - 2
        return len(self.shape)

    @property
    def shape(self):
        """return shape of phase array"""
        # return self._collection.shape[:-1]
        return self._collection.shape

    @property
    def size(self):
        """return size of phase array"""
        return np.prod(self.states.shape[:-2])

    @property
    def nstate(self):
        """return number of phase states"""
        # return (self._collection.shape[-1] - 1) // 2
        return (self.states.shape[-2] - 1) // 2

    @property
    def kdim(self):
        """number of coords dimensions"""
        return 1 if self.coords is None else self.coords.shape[-1]

    @property
    def i0(self):
        """index/indices of F0 state(s)"""
        if self.kdim < 4:
            return self.nstate
        xp = common.get_array_module(self.states)
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
        """wavenumbers (only first 3 dimensions)"""
        coords = _setup_coords(self.nstate, 1) if self.coords is None else self.coords
        return coords[..., :3] * self.kvalue

    @property
    def t(self):
        """time-accumulated dephasing (4th dimension)"""
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
        coeff = [self.kvalue] * min(self.kdim, 3) + [self.tvalue] * (self.kdim == 4)
        return np.array(coeff)

    @property
    def norm(self):
        """state matrix norm"""
        return np.sqrt(np.sum(np.abs(self.states[..., 1:]) ** 2, axis=(-2, -1)))

    @property
    def zeros(self):
        """zero state matrix with current shape/nstate"""
        return self.copy([0, 0, 0], nstate=self.nstate, shape=self.shape, check=False)

    @property
    def writeable(self):
        return getattr(self.states.flags, "writeable", False)

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
        elif np.isscalar(other):
            value = other
        else:  # array
            value = xp.asarray(other)
        return value

    def __add__(self, other):
        return self.copy(self.states + self._cmp(other), check=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        self.states[:] += self._cmp(other)
        return self

    def __mul__(self, other):
        return self.copy(self.states * self._cmp(other), check=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        self.states[:] *= self._cmp(other)
        return self

    def __eq__(self, other):
        return self.states == self._cmp(other)

    # public functions

    def copy(self, states=None, *, check=False, **kwargs):
        """copy state matrix"""
        states = states if not states is None else self.states
        kwargs = {
            "check": check,
            "kvalue": self.kvalue,
            "tvalue": self.tvalue,
            **self.options,
            **kwargs,
        }
        cls = type(self)
        sm = cls(states, equilibrium=self.equilibrium, coords=self.coords, **kwargs)
        return sm

    def resize(self, nstate):
        """resize state matrix to nstate"""
        if nstate == self.nstate:
            return
        # self._collection.resize_axis(2 * nstate + 1, -1)
        self._collection.resize("nstate", 2 * nstate + 1)

    def expand(self, ndim):
        """expend state matrix to n-dimensions"""
        # self._collection.expand(ndim + 1, insert_index=-1)
        diff = ndim - self._collection.ndim
        if diff > 0:
            self._collection.expand(diff)

    def reduce(self, ndim):
        """expend state matrix to n-dimensions"""
        # self._collection.reduce(ndim + 1, remove_index=-1)
        diff = self._collection.ndim - ndim
        if diff > 0:
            self._collection.reduce(diff)

    def check(self):
        return utils.check_states(self.states)

    def setup_coords(self, kdim):
        if self.coords is not None:
            # resize
            diff = kdim - self.kdim
            if diff > 0:
                xp = self._collection.xp
                coords = self.coords
                zeros = xp.zeros(coords.shape[:-1] + (diff,))
                coords = xp.concatenate([coords, zeros], axis=-1)
            elif diff < 0:
                raise RuntimeError("Cannot remove existimg k-dimension")
            else:
                return
        else:
            coords = _setup_coords(self.nstate, kdim)
        # self._collection.set("coords", coords)
        self._collection.set("coords", coords, layout=[..., "nstate", "kdim"])


# private functions


def _init_states(density=1):
    xp = common.get_array_module()
    density = xp.atleast_1d(density)
    density = xp.expand_dims(density, [density.ndim, density.ndim + 1])
    init = xp.array([[[0, 0, 1]]])
    states = density * init
    return states


def _format_states(states, check=True):
    """format and check states matrix"""
    # init state matrix
    states = common.asarray(states).astype(np.complex128)

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
        if not np.allclose(states[..., 1], states[..., ::-1, 0].conj()):
            raise ValueError("The F-state columns do no match.")
        if not np.allclose(states[..., 2], states[..., ::-1, 2].conj()):
            raise ValueError("The Z-state columns is not symmetrical.")
    return states


def _setup_coords(nstate, kdim=1):
    """setup wavenumbers array"""
    xp = common.get_array_module()
    coords = np.c_[
        xp.arange(-nstate, nstate + 1), xp.zeros((2 * nstate + 1, kdim - 1), dtype=int)
    ]
    coords = common.expand_dims(coords, (0,))
    return coords


#
# array collection


class ArrayCollection:
    """Broadcastable arrays

    Each array has a layout whose items define the broadcastable structure:
    Items:
        None: free axis dimension
        <int>: fixed axis dimension
        Ellipsis: broadcastable axes
        <str>: shared axis dimension

    Example:
        # whole array is broadcast
        coll.set('arr1', arr1, [Ellipsis])
        # last axis of arr2 is free
        coll.set('arr2', arr2, [Ellipsis, None])
        # first axis of arr3 is fixed with size==3
        coll.set('arr3', arr3, [3, Ellipsis])
        # first axis of arr4 and last axis of arr4 are shared
        coll.set('arr4', arr4, ['ax', Ellipsis])
        coll.set('arr5', arr5, [Ellipsis, 'ax'])

    """

    def __init__(self, default=None, *, expand_axis=0):
        self._expand_axis = expand_axis
        self._layouts = {}
        self._arrays = {}
        # default shape
        self._default = (1,) if default is None else tuple(default)
        self.xp = common.get_array_module()

    def __len__(self):
        return len(self._arrays)

    def __repr__(self):
        return f"ArrayCollection({len(self._arrays)})"

    def __iter__(self):
        return iter(self._arrays)

    def __contains__(self, name):
        return name in self._arrays

    @property
    def ndim(self):
        """broadcast ndim"""
        layouts = [self._layouts[name] for name in self] + [(Ellipsis,)]
        ndims = [self._arrays[name].ndim for name in self] + [len(self._default)]
        ndims = [ndim - len(ax) + 1 for ax, ndim in zip(layouts, ndims)]
        return max(ndims + [0])

    @property
    def expand_axis(self):
        ax = self._expand_axis
        return ax if ax >= 0 else self.ndim + ax + 1

    @property
    def shape(self):
        """broadcast shape"""
        layouts = [self._layouts[name] for name in self] + [(Ellipsis,)]
        indices = [list(layout).index(Ellipsis) for layout in layouts]
        shapes = [self._arrays[name].shape for name in self] + [self._default]
        shapes = [
            shape[idx : idx + len(shape) - len(layout) + 1]
            for idx, layout, shape in zip(indices, layouts, shapes)
        ]
        # expand shapes
        insert = self.expand_axis
        ndim = self.ndim
        shapes = [
            shape[:insert] + (1,) * (ndim - len(shape)) + shape[insert:]
            for shape in shapes
        ]
        # return broadcast shape
        return tuple(max(shape[i] for shape in shapes) for i in range(ndim))

    @property
    def axes(self):
        """dictionary of named axes"""
        return self.get_axes()

    def get(self, name, default=None):
        if not name in self._arrays:
            return default
        layout = self._layouts[name]
        array = self._arrays[name]
        return self._broadcast_array(array, layout)

    def set(self, name, array, *, layout=None):
        """layout = (..., None, 3, 'a')"""
        xp = self.xp
        array = xp.array(array)  # copy array

        if name in self._arrays:
            ignore = name
            if layout is None:
                # keep existing layout
                layout = self._layouts[name]
        elif layout is None:
            layout = [Ellipsis]

        self.check(array.shape, layout, ignore=name)
        self._arrays[name] = array
        self._layouts[name] = layout
        
            
    def pop(self, name, default=None):
        if not name in self._arrays:
            return default
        self._layouts.pop(name)
        array = self._arrays.pop(name)
        return array

    # utilities
    def get_axes(self, ignore=None):
        axes = {}
        for name in self:
            if name == ignore:
                continue
            array = self._arrays[name]
            layout = self._layouts[name]
            idx = 0
            for ax in layout:
                if ax is Ellipsis:
                    idx += array.ndim - len(layout) + 1
                    continue
                elif isinstance(ax, str):
                    axes.setdefault(ax, array.shape[idx])
                idx += 1
        return axes

    def check(self, shape, layout=(...,), *, ignore=None):
        """check shape and layout"""
        shape = tuple(shape)
        axes = self.get_axes(ignore=ignore)
        axis, idx = None, 0
        # check layout
        for i, ax in enumerate(layout):
            if ax is Ellipsis:
                if axis is not None:
                    raise ValueError(f"`layout` must contain one Ellipsis: {layout}")
                axis = i
                idx += len(shape) - len(layout) + 1
                continue
            elif isinstance(ax, int) and shape[idx] != ax:
                raise ValueError(f"Invalid axis dimension {idx} in array: {shape}")
            elif isinstance(ax, str) and shape[idx] != axes.get(ax, shape[idx]):
                raise ValueError(
                    f"Invalid named axis dimension {ax}={shape[idx]} in array: {shape}"
                )
            idx += 1
        if axis is None:
            raise ValueError(f"`layout` must contain one Ellipsis: {layout}")

        # check shape
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

    def resize(self, ax, size, *, constant=0):
        """resize named axis"""
        xp = self.xp
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
            if diff < 0:  # crop
                slices = [slice(None) for _ in range(arr.ndim)]
                slices[axis] = slice(-diff // 2, arr.shape[axis] - (-diff + 1) // 2)
                arr = arr[tuple(slices)]
            elif diff > 0:  # padd
                pad = [(0, 0) for _ in range(arr.ndim)]
                pad[axis] = (diff // 2, (diff + 1) // 2)
                arr = xp.pad(arr, pad, constant_values=constant)
            self._arrays[name] = arr

    def expand(self, ndim):
        """add dimensions to broadcast shape"""
        shape = self._default
        axis = self.expand_axis
        ndim += self.ndim - len(shape)
        shape = shape[axis:] + (1,) * ndim + shape[:axis]
        self._default = tuple(shape)

    def broadcast(self, shape):
        """broadcast collection to shape"""
        self.check(shape, [...])
        self._default = shape

    def reduce(self, ndim, *, axis=0):
        """remove dimensions from default shape"""
        shape = list(self._default)
        axis = self._expand_axis
        if axis >= 0:
            shape[axis : axis + ndim] = []
        else:
            axis = len(shape) + axis + 1
            shape[axis - ndim : axis + 1] = []
        self._default = tuple(shape)

    # private

    def _broadcast_array(self, array, layout):
        """expand and broadcast array to shared shape"""
        xp = self.xp
        self.check(array.shape, layout)

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
        if tuple(shape) != array.shape:
            array = xp.broadcast_to(array, shape)
        return array
