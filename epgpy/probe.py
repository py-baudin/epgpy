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
        locals = common.DeferredGetter(sm, self.SM_LOCALS)
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
            attr: str, StateMatrix attribute to probe (F0, F, Z0, etc.)
            phase: [None]/ndarray, ADC phase compensation *in degrees*
                Note: this is applied *after* reduction (cf. `reduce` below)
            reduce: [None]/True/False/int/tuple[int], add output array along given axis/axes
                If True: add along all axes; if False: do not reduce
            weights: [None]/ndarray, array of weights to apply
                If `reduce` is None, it is automatically set to all `weights` axes
        """
        if not attr in self.SM_LOCALS:
            raise ValueError(f"Invalid StateMatrix attribute: {attr}")
        self.attr = attr

        # phase compensation
        if phase is not None:
            phrepr = common.repr_value(phase, ".1f")
            self._repr = f"'{attr}', {phrepr}"
            phase = np.asarray(phase)
            self.phasor = np.exp(1j * phase / 180 * np.pi)
        else:
            self._repr = attr
        self.phase = phase

        # reduce axes
        if reduce is not None:
            if reduce is True:
                pass
            elif reduce:
                reduce = (reduce,) if isinstance(reduce, int) else tuple(reduce)
                if not all(isinstance(ax, int) for ax in reduce):
                    raise ValueError(f"Expected (tuple of) int, got: {reduce}")
        self.reduce = reduce

        # reduction weights
        if weights is not None:
            weights = np.asarray(weights)
            ndim = max(weights.ndim, 1)
            if reduce is None:
                # reduce along all weights axes
                self.reduce = tuple(range(ndim))
            elif reduce is True:
                # reduce all axes
                pass
            elif reduce:
                # check reduce axes
                if not set(reduce) <= set(range(ndim)):
                    raise ValueError(f"Invalid reduce dimension(s): {reduce}")
        self.weights = weights
        operator.Operator.__init__(self, name=name)

    def _acquire(self, sm):
        arr = getattr(sm, self.attr)
        # weights
        if self.weights is not None:
            weights = self.weights
            if weights.size > 1 and weights.ndim < arr.ndim:
                dims = tuple(range(weights.ndim, arr.ndim))
                weights = np.expand_dims(weights, dims)
            arr = arr * weights
        # reduce
        if self.reduce is None or self.reduce is False:
            return arr
        elif self.reduce is True:
            return arr.sum()
        # else:
        return arr.sum(axis=self.reduce)

    def _post(self, obj):
        """apply phase compensation, weights and reduction"""
        arr = np.asarray(obj)
        # phase
        if self.phase is not None:
            phasor = self.phasor
            if phasor.size > 1 and phasor.ndim < arr.ndim:
                dims = tuple(range(phasor.ndim, arr.ndim))
                phasor = np.expand_dims(phasor, dims)
            arr = arr * phasor
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
