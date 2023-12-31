import numpy as np
from . import operator, common, utils

NAX = np.newaxis


class Probe(operator.EmptyOperator):
    """Does nothing, holds callback function to record data"""

    # StateMatrix attributes accessible in 'eval'
    SM_LOCALS = [
        "nstate",
        "ndim",
        "kdim",
        "states",
        "coords",
        "F",
        "F0",
        "Z",
        "Z0",
        "k",
        "t",
        "t0",
    ]

    # phase compensation value, used by the simulate function
    phase = 0

    def __init__(self, obj, *args, post=None, **kwargs):
        """Init Probe operator with callable or eval expression.

        Args:
            obj:
            - if callable: first argument is the state matrix
            - if str: eval expression, cf. list of locals: self.SM_LOCALS
        """

        if isinstance(obj, str):
            # eval expression
            def acquire(sm):
                # expose StateMatrix attributes only if accessed
                locals = utils.DeferredGetter(sm, self.SM_LOCALS)
                locals.update(kwargs)
                return eval(obj, vars(np), locals)
                # return np.copy(eval(obj, vars(np), locals))

        elif callable(obj):
            # callable
            def acquire(sm):
                return obj(sm, *args)
                # return np.copy(obj(sm, *args))

        # probe function
        self._acquire = acquire
        self._post = post
        self._repr = f"'{obj}'"

        # init parent class
        super().__init__()

    def acquire(self, sm, post=None):
        """return object (copy onto CPU)"""
        post = post if post else self.post
        return post(common.asnumpy(self._acquire(sm), copy=True))

    def post(self, obj):
        """post process acquired object"""
        if not getattr(self, "_post", None):
            return obj
        return self._post(obj)

    def __call__(self, sm, **kwargs):
        """just pass the state matrix"""
        return sm

    def __repr__(self):
        return f"Probe({self._repr})"


class Adc(Probe):
    """Simplified Probe operator, with phase compensation argument"""

    def __init__(self, attr="F0", *, phase=None, reduce=None, weights=None, name="ADC"):
        """Init ADC operator

        Args:
            attr: StateMatrix attribute to probe (F0, F, Z0, etc.)
            phase: phase compensation *in degrees*
                Used by `simulate` function *if ADC object inserted the sequence*
        """
        if not attr in self.SM_LOCALS:
            raise ValueError(f"Invalid StateMatrix attribute: {attr}")
        self.attr = attr
        fmts = {"tau": ".1f", "T1": ".1f", "T2": ".1f", "g": ".3f"}

        # multiplier
        self._mult = 1

        # phase compensation
        self.phase = np.asarray(phase)
        if phase is not None:
            phrepr = common.repr_value(phase, ".1f")
            self._repr = f"'{attr}', {phrepr}"
            self._mult = np.exp(1j * self.phase / 180 * np.pi)
        else:
            self._repr = attr

        # reduction weights
        ndim = max(len(np.shape(weights)), 1)
        if weights is not None:
            self._mult *= np.asarray(weights)
            if reduce is None:
                # reduce along all weights axes
                reduce = tuple(range(ndim))
            elif reduce is not Ellipsis:
                reduce = (reduce,) if np.isscalar(reduce) else tuple(reduce)
                if not set(reduce) <= set(range(ndim)):
                    raise ValueError(f"Invalid reduce dimension(s): {reduce}")
        self.weights = weights
        self.reduce = reduce
        operator.Operator.__init__(self, name=name)

    def _acquire(self, sm):
        return getattr(sm, self.attr)

    def _post(self, obj):
        """apply phase compensation, weights and reduction"""
        arr = np.asarray(obj)
        mult = self._mult
        if not np.isscalar(mult):
            dims = [i for i in range(arr.ndim) if i >= mult.ndim]
            mult = np.expand_dims(mult, dims)
        arr = obj * mult
        if self.reduce not in [None, Ellipsis]:
            arr = np.sum(arr, axis=self.reduce)
        return arr


class Imaging(Probe):
    """Imaging ADC

    Discrete Fourier transform of the F-states
    """

    def __init__(self, pos, *, name=None, expand=False, reduce=None):
        self.xp = common.get_array_module()
        pos = common.asarray(pos)
        self.pos = pos[..., NAX] if pos.ndim == 1 else pos
        self.kdim = self.pos.shape[-1]
        self.expand = expand
        self.reduce = reduce
        self._repr = "Imaging"
        operator.Operator.__init__(self, name=name, duration=None)

    def _acquire(self, sm):
        # discrete Fourier transform
        return dft(
            sm.F,
            sm.k[..., : self.kdim],
            self.pos,
            expand=self.expand,
            reduce=self.reduce,
        )


# functions
def dft(states, wavenumbers, positions, *, expand=False, reduce=None):
    """Discrete Fourier transform

    Args:
        states:         ... x nstate
        wavenumbers:    ... x nstate x ndim
        positions:      ... x ndim

    """
    xp = common.get_array_module(states)
    F = common.asarray(states)
    k = common.asarray(wavenumbers)
    pos = common.asarray(positions)
    pos = pos if pos.ndim > 1 else pos[..., NAX]
    if expand:
        # insert pos dimensions into F and k
        dims = np.arange(pos.ndim - 1)
        k = xp.expand_dims(k, tuple(-3 - dims))
        F = xp.expand_dims(F, tuple(-2 - dims))
    s = xp.sum(F * xp.exp(1j * xp.einsum("...ni,...i->...n", k, pos)), axis=-1)
    if reduce is not None:
        s = np.sum(s, axis=reduce)
    return s


# ADC instance (returns F0)
ADC = Adc(attr="F0", name="ADC")
