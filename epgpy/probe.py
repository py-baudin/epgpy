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
            self._expr = obj
            self._acquire = self._acquire_expr
        elif callable(obj):
            self._callable = obj
            self._acquire = self._acquire_callable

        self._args = args
        self._kwargs = kwargs

        # probe function

        self._post = post
        self._repr = f"'{obj}'"

        # init parent class
        super().__init__()

    def _acquire_expr(self, sm):
        # expose StateMatrix attributes only if accessed
        locals = utils.DeferredGetter(sm, self.SM_LOCALS)
        locals.update(self._kwargs)
        return eval(self._expr, vars(np), locals)

    def _acquire_callable(self, sm):
        return self._callable(sm, *self._args, **self._kwargs)

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
        return self.name or f"Probe({self._repr})"


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
            self._mult = self._mult * np.asarray(weights)
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
        if self.reduce is True:
            arr = np.sum(arr)
        elif self.reduce not in [None, Ellipsis]:
            arr = np.sum(arr, axis=self.reduce)
        return arr


class DFT(Probe):
    """Discrete Fourier transform"""

    def __init__(self, coords=None, *, name=None):
        if coords is not None:
            xp = common.get_array_module()
            coords = xp.asarray(coords)
        self.coords = coords
        self._repr = "DFT"
        operator.Operator.__init__(self, name=name, duration=None)

    def _acquire(self, sm):
        coords = self.coords if self.coords is not None else sm.system["coords"]
        return utils.dft(coords, sm.F, sm.k[..., :3])


class Imaging(Probe):
    """Imaging ADC

    Discrete Fourier transform of the F-states
    """

    def __init__(self, coords=None, *, name=None, **opts):
        if coords is not None:
            xp = common.get_array_module()
            coords = xp.asarray(coords)
        self.coords = coords
        self._repr = "Imaging"
        self.opts = opts
        operator.Operator.__init__(self, name=name, duration=None)

    def _acquire(self, sm):
        # get corods, modultion and weights from attribute or sm.system
        coords = self.coords
        if coords is None:
            coords = sm.system.get("coords", broadcast=False)
        modulation = self.opts.pop("modulation", None)
        if modulation is None:
            modulation = sm.system.get("modulation", broadcast=False)
        weights = self.opts.pop("weights", None)
        if weights is None:
            weights = sm.system.get("weights", broadcast=False)
        # imaging function
        return utils.imaging(
            coords,
            sm.F,
            sm.k[..., :3],
            acctime=sm.t if sm.kdim == 4 else None,
            modulation=modulation,
            weights=weights,
            **self.opts,
        )


# ADC instance (returns F0)
ADC = Adc(attr="F0", name="ADC")
