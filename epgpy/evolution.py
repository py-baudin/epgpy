"""Evolution functions"""

import numpy as np

# from . import bloch, common
from . import common, opscalar


class R(opscalar.ScalarOp):
    """n-dimensional evolution operator"""

    PARAMETERS_ORDER1 = {"rT", "rL", "r0"}
    PARAMETERS_ORDER2 = {("rT", "rT"), ("rL", "rL"), ("r0", "r0")}

    def __init__(
        self, rT=0, rL=0, *, r0=None, axes=None, name=None, duration=None, **kwargs
    ):
        """Initialize evolution operator with relaxation/precession and recovery arrays

        Args:
            axes: shift axes to given indices
            cf. Operator for other arguments

        """
        rT, rL, r0 = common.map_arrays([rT, rL, r0])

        if not name:  # default name
            name = common.repr_operator(
                "R",
                ["rT", "rL", "r0"],
                [rT, rL, r0],
                [".1f", ".1f", ".1f"],
            )

        # store parameters
        self.rT = rT
        self.rL = rL
        self.r0 = r0

        # init operator
        opscalar.diff.DiffOperator.__init__(
            self, name=name, duration=duration, **kwargs
        )

        # init arrays
        arr, arr0 = evolution_operator(rT, rL, r0)

        # derivatives
        darrs, d2arrs = {}, {}
        order1, order2 = self.parameters_order1, self.parameters_order2
        if order1 or order2:
            # derivatives
            if "rT" in order1:
                darrs["rT"] = evolution_d_rT(rT, rL, r0)
            if "rL" in order1:
                darrs["rL"] = evolution_d_rL(rT, rL, r0)
            if "r0" in order1:
                darrs["r0"] = evolution_d_r0(rT, rL, r0)
            if ("rT", "rT") in order2:
                d2arrs[("rT", "rT")] = evolution_d2_rT(rT, rL, r0)
            if ("rL", "rL") in order2:
                d2arrs[("rL", "rL")] = evolution_d2_rL(rT, rL, r0)
            if ("r0", "r0") in order2:
                d2arrs[("r0", "r0")] = evolution_d2_r0(rT, rL, r0)

        self._init(arr, arr0, darrs=darrs, d2arrs=d2arrs, axes=axes)


class E(opscalar.ScalarOp):
    """n-dimensional evolution operator"""

    PARAMETERS_ORDER1 = {"tau", "T1", "T2", "g"}
    PARAMETERS_ORDER2 = {
        ("tau", "tau"),
        ("T1", "T1"),
        ("T2", "T2"),
        ("g", "g"),
        ("T1", "tau"),
        ("T2", "tau"),
        ("g", "tau"),
        ("T2", "g"),
    }

    def __init__(
        self, tau, T1, T2, g=0, *, axes=None, name=None, duration=None, **kwargs
    ):
        """Initialize evolution operator with relaxation and precession rates

        Args:
            tau: evolution time (ms)
            T1: longitudinal relaxation time (ms)
            T2: transverse relaxation time (ms)
            g: precession rate (kHz)
            axes: shift axes to given indices
            cf. Operator for other arguments

        """
        tau, T1, T2, g = common.map_arrays([tau, T1, T2, g])

        if not name:  # default name
            name = common.repr_operator(
                "E",
                ["tau", "T1", "T2", "g"],
                [tau, T1, T2, g],
                [".1f", ".1f", ".1f", ".3f"],
            )
        self.tau = tau
        self.T1 = T1
        self.T2 = T2
        self.g = g

        # match duration and tau if `duration` is `True`
        self._duration = duration
        duration = self.tau if duration is True else duration

        # init operator
        opscalar.diff.DiffOperator.__init__(
            self, name=name, duration=duration, **kwargs
        )

        # init arrays
        arr, arr0 = relaxation_operator(tau, T1, T2, g)

        # derivatives
        darrs, d2arrs = {}, {}
        order1, order2 = self.parameters_order1, self.parameters_order2
        if order1 or order2:
            if "tau" in order1:
                darrs["tau"] = relaxation_d_tau(tau, T1, T2, g)
            if "T1" in order1:
                darrs["T1"] = relaxation_d_T1(tau, T1, T2, g)
            if "T2" in order1:
                darrs["T2"] = relaxation_d_T2(tau, T1, T2, g)
            if "g" in order1:
                darrs["g"] = relaxation_d_g(tau, T1, T2, g)
            if ("tau", "tau") in order2:
                d2arrs[("tau", "tau")] = relaxation_d2_tau(tau, T1, T2, g)
            if ("T1", "T1") in order2:
                d2arrs[("T1", "T1")] = relaxation_d2_T1(tau, T1, T2, g)
            if ("T2", "T2") in order2:
                d2arrs[("T2", "T2")] = relaxation_d2_T2(tau, T1, T2, g)
            if ("g", "g") in order2:
                d2arrs[("g", "g")] = relaxation_d2_g(tau, T1, T2, g)
            if ("T1", "tau") in order2:
                d2arrs[("T1", "tau")] = relaxation_d_tau_T1(tau, T1, T2, g)
            if ("T2", "tau") in order2:
                d2arrs[("T2", "tau")] = relaxation_d_tau_T2(tau, T1, T2, g)
            if ("g", "tau") in order2:
                d2arrs[("g", "tau")] = relaxation_d_tau_g(tau, T1, T2, g)
            if ("T2", "g") in order2:
                d2arrs[("T2", "g")] = relaxation_d_T2_g(tau, T1, T2, g)

        self._init(arr, arr0, darrs=darrs, d2arrs=d2arrs, axes=axes)


class P(opscalar.ScalarOp):
    """n-dimensional evolution operator"""

    PARAMETERS_ORDER1 = {"tau", "g"}
    PARAMETERS_ORDER2 = {("tau", "tau"), ("g", "g"), ("g", "tau")}

    def __init__(self, tau, g, *, axes=None, name=None, duration=None, **kwargs):
        """Initialize evolution operator with precession rate

        Args:
            tau: evolution time (ms)
            g: precession rate (kHz)
            axes: shift axes to given indices
            cf. Operator for other arguments

        """
        tau, g = common.map_arrays([tau, g])

        if not name:  # default name
            name = common.repr_operator(
                "P",
                ["tau", "g"],
                [tau, g],
                [".1f", ".3f"],
            )

        # store parameters
        self.tau = tau
        self.g = g

        # match duration and tau if `duration` is `True`
        self._duration = duration
        duration = self.tau if duration is True else duration

        # init operator
        opscalar.diff.DiffOperator.__init__(
            self, name=name, duration=duration, **kwargs
        )

        # init arrays
        arr, arr0 = precession_operator(tau, g)

        # derivatives
        darrs, d2arrs = {}, {}
        order1, order2 = self.parameters_order1, self.parameters_order2
        if order1 or order2:
            if "tau" in order1:
                darrs["tau"] = precession_d_tau(tau, g)
            if "g" in order1:
                darrs["g"] = precession_d_g(tau, g)
            if ("tau", "tau") in order2:
                d2arrs[("tau", "tau")] = precession_d2_tau(tau, g)
            if ("g", "g") in order2:
                d2arrs[("g", "g")] = precession_d2_g(tau, g)
            if ("g", "tau") in order2:
                d2arrs[("g", "tau")] = precession_d_tau_g(tau, g)

        self._init(arr, arr0, darrs=darrs, d2arrs=d2arrs, axes=axes)


#
# functions


def evolution_operator(rT, rL, r0=None):
    """return evolution operator"""
    rT, rL, r0 = common.expand_arrays(rT, rL, r0, append=True)
    shape = common.broadcast_shapes(
        common.get_shape(rT),
        common.get_shape(rL),
        common.get_shape(r0),
        [1],  # minimum shape
    )

    # evolution matrix
    mat = np.zeros(shape + (3,), dtype=np.complex128)
    mat[..., 1] = np.exp(-rT)
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = np.exp(-rL)

    if r0 is not None:
        # recovery
        mat0 = np.zeros(shape + (3,), dtype=np.complex128)
        mat0[..., 2] = 1 - np.exp(-r0)
    else:
        mat0 = None
    return mat, mat0


def precession_operator(tau, g):
    tau, g = common.expand_arrays(tau, g, append=True)
    rT = 2j * np.pi * g * tau
    return evolution_operator(rT, rL=0, r0=None)


def relaxation_operator(tau, T1, T2, g):
    """return evolution operator"""
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    return evolution_operator(rT, rL, rL)


#
# derivatives


def evolution_d_rT(rT, rL, r0=None):
    mat, _ = evolution_operator(rT, 0)
    mat[..., 2] = 0
    return -mat, None


def evolution_d_rL(rT, rL, r0=None):
    mat, _ = evolution_operator(0, rL)
    mat[..., :-1] = 0
    return -mat, None


def evolution_d_r0(rT, rL, r0=None):
    assert r0 is not None, "r0 cannot be None"
    mat, mat0 = evolution_operator(0, 0, r0)
    mat[:] = 0
    mat0[..., -1] -= 1
    return mat, -mat0


def evolution_d2_rT(rT, rL, r0=None):
    mat, _ = evolution_operator(rT, 0)
    mat[..., 2] = 0
    return mat, None


def evolution_d2_rL(rT, rL, r0=None):
    mat, _ = evolution_operator(0, rL)
    mat[..., :-1] = 0
    return mat, None


def evolution_d2_r0(rT, rL, r0=None):
    assert r0 is not None, "r0 cannot be None"
    mat, mat0 = evolution_operator(0, 0, r0)
    mat[:] = 0
    mat0[..., -1] -= 1
    return mat, mat0


def evolution_d_cross(rT, rL, r0=None):
    mat, _ = evolution_operator(rT, 0)
    mat[:] *= 0
    return mat, None


#
# precession


def precession_d_tau(tau, g):
    rT = 2j * np.pi * g * tau
    mat, _ = evolution_operator(rT, rL=0, r0=None)
    mat[..., 1] *= -2j * np.pi * g
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def precession_d_g(tau, g):
    rT = 2j * np.pi * g * tau
    mat, _ = evolution_operator(rT, rL=0, r0=None)
    mat[..., 1] *= -2j * np.pi * tau
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def precession_d2_tau(tau, g):
    rT = 2j * np.pi * g * tau
    mat, _ = evolution_operator(rT, rL=0, r0=None)
    mat[..., 1] *= (-2j * np.pi * g) ** 2
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def precession_d2_g(tau, g):
    rT = 2j * np.pi * g * tau
    mat, _ = evolution_operator(rT, rL=0, r0=None)
    mat[..., 1] *= (-2j * np.pi * tau) ** 2
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def precession_d_tau_g(tau, g):
    rT = 2j * np.pi * g * tau
    mat, _ = evolution_operator(rT, rL=0, r0=None)
    mat[..., 1] *= -2j * np.pi * (1 - 2j * np.pi * g * tau)
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


#
# Relaxation
def relaxation_d_tau(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    mat, mat0 = evolution_operator(rT, rL, rL)
    mat[..., 1] *= -rT / tau
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] *= -1 / T1
    mat0[..., 2] = -mat[..., 2]
    return mat, mat0


def relaxation_d_T1(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rL = tau / T1
    mat, mat0 = evolution_operator(0, rL, rL)
    mat[..., :2] = 0
    mat[..., 2] *= tau / T1**2
    mat0[..., 2] = -mat[..., 2]
    return mat, mat0


def relaxation_d_T2(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    mat, _ = evolution_operator(rT, 0, None)
    mat[..., 0] *= tau / T2**2
    mat[..., 1] *= tau / T2**2
    mat[..., 2] = 0
    return mat, None


def relaxation_d_g(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    mat, _ = evolution_operator(rT, 0, None)
    mat[..., 1] *= -2j * np.pi * tau
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


# order2


def relaxation_d2_tau(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    mat, mat0 = evolution_operator(rT, rL, rL)
    mat[..., 1] *= (rT / tau) ** 2
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] *= 1 / T1**2
    mat0[..., 2] = -mat[..., 2]
    return mat, mat0


def relaxation_d2_T1(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rL = tau / T1
    mat, mat0 = evolution_operator(0, rL, rL)
    mat[..., :2] = 0
    mat[..., 2] *= tau**2 / T1**4 - 2 * tau / T1**3
    mat0[..., 2] = -mat[..., 2]
    return mat, mat0


def relaxation_d2_T2(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    mat, _ = evolution_operator(rT, 0, None)
    mat[..., 0] *= tau**2 / T2**4 - 2 * tau / T2**3
    mat[..., 1] *= tau**2 / T2**4 - 2 * tau / T2**3
    mat[..., 2] = 0
    return mat, None


def relaxation_d2_g(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    mat, _ = evolution_operator(rT, 0, None)
    mat[..., 1] *= (-2j * np.pi * tau) ** 2
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def relaxation_d_tau_T1(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    mat, mat0 = evolution_operator(rT, rL, rL)
    mat[..., :2] = 0
    mat[..., 2] *= (1 - rL) / T1**2
    mat0[..., 2] = -mat[..., 2]
    return mat, mat0


def relaxation_d_tau_T2(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    mat, _ = evolution_operator(rT, rL, rL)
    mat[..., 1] *= (1 - rT) / T2**2
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def relaxation_d_tau_g(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    mat, _ = evolution_operator(rT, rL, rL)
    mat[..., 1] *= -2j * np.pi * (1 - rT)
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None


def relaxation_d_T2_g(tau, T1, T2, g=0):
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, append=True)
    rT = tau * (1 / T2 + 2j * np.pi * g)
    rL = tau / T1
    mat, _ = evolution_operator(rT, rL, rL)
    mat[..., 1] *= -2j * np.pi * (tau / T2) ** 2
    mat[..., 0] = mat[..., 1].conj()
    mat[..., 2] = 0
    return mat, None
