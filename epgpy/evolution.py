""" Evolution functions """
import numpy as np

# from . import bloch, common
from . import linalgebra, common


class R(linalgebra.ScalarOp):
    """n-dimensional evolution operator"""

    def __init__(self, rT=0, rL=0, *, r0=None, axes=None, name=None, duration=None):
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

        # match duration and tau if `duration` is `True`
        self._duration = self.tau if duration is True else duration

        # coefficients
        coeff, coeff0 = evolution_operator(rT, rL, r0)

        # init operator
        super().__init__(coeff, coeff0, axes=axes, name=name, duration=duration)


class E(linalgebra.ScalarOp):
    """n-dimensional evolution operator"""

    def __init__(self, tau, T1, T2, g=0, *, axes=None, name=None, duration=None):
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
        self._duration = self.tau if duration is True else duration

        # coefficients
        coeff, coeff0 = relaxation_operator(tau, T1, T2, g)

        super().__init__(coeff, coeff0, axes=axes, name=name, duration=duration)


class P(linalgebra.ScalarOp):
    """n-dimensional evolution operator"""

    def __init__(self, tau, g, *, axes=None, name=None, duration=None):
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
        self._duration = self.tau if duration is True else duration

        # coefficients
        coeff, coeff0 = precession_operator(tau, g)

        # init operator
        super().__init__(coeff, coeff0, axes=axes, name=name, duration=duration)


#
# functions


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


# def _evolution(tau, T1, T2, g):
#     """T1/T2 decay and g precession"""
#     # initialize matrices with numpy
#     xp = np

#     # expand arrays
#     tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g, prepend=True)

#     if T1 is not None:
#         E1 = xp.exp(-tau / T1)
#     else:
#         E1 = 1

#     if T2 is not None:
#         E2 = xp.exp(-tau / T2)
#     else:
#         E2 = 1

#     if g is not None:
#         P = xp.exp(tau * 2j * xp.pi * g)
#     else:
#         P = 1

#     shape = common.broadcast_shape(
#         common.get_shape(tau),
#         common.get_shape(T1),
#         common.get_shape(T2),
#         common.get_shape(g),
#         [1],  # minimum shape: (1,)
#     )
#     decay = xp.zeros(shape + (3, 3), dtype=xp.complex128)
#     decay[..., 0, 0] = E2 * P
#     decay[..., 1, 1] = E2 * xp.conj(P)
#     decay[..., 2, 2] = E1

#     # T1 recovery
#     recovery = xp.zeros(shape + (3, 3), dtype=xp.complex128)
#     recovery[..., 2, 2] = 1 - E1

#     return decay, recovery


#
# derivatives


def evolution_d_tau(tau, T1, T2, g):
    """gradient of relax matrix w/r tau"""
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE1 = -1 / T1
    dE2 = -1 / T2 + 2j * xp.pi * g
    decay[..., 0, 0] *= dE2
    decay[..., 1, 1] *= xp.conj(dE2)
    decay[..., 2, 2] *= dE1
    recovery[..., 2, 2] = -dE1 * xp.exp(-tau / T1)

    return decay, recovery


def evolution_d_T1(tau, T1, T2, g):
    """gradient of relax matrix w/r T1"""
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE1 = tau / T1**2
    decay[..., 0, 0] = 0
    decay[..., 1, 1] = 0
    decay[..., 2, 2] *= dE1
    recovery[..., 2, 2] = -dE1 * xp.exp(-tau / T1)
    return decay, recovery


def evolution_d_T2(tau, T1, T2, g):
    """gradient of relax matrix w/r T2"""
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE2 = tau / T2**2
    decay[..., 0, 0] *= dE2
    decay[..., 1, 1] *= dE2
    decay[..., 2, 2] = 0
    recovery *= 0
    return decay, recovery


def evolution_d_g(tau, T1, T2, g):
    """gradient of relax matrix w/r g (precession frequency in kHz)"""
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dg = 2j * xp.pi * tau
    decay[..., 0, 0] *= dg
    decay[..., 1, 1] *= -dg
    decay[..., 2, 2] = 0
    recovery *= 0
    return decay, recovery


def evolution_d2_tau(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE1 = (-1 / T1) ** 2
    dE2 = (-1 / T2 + 2j * xp.pi * g) ** 2
    decay[..., 0, 0] *= dE2
    decay[..., 1, 1] *= xp.conj(dE2)
    decay[..., 2, 2] *= dE1
    recovery[..., 2, 2] = -dE1 * xp.exp(-tau / T1)
    return decay, recovery


def evolution_d2_T1(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE1 = tau / T1**3 * (tau / T1 - 2)
    decay[..., 0, 0] = 0
    decay[..., 1, 1] = 0
    decay[..., 2, 2] *= dE1
    recovery[..., 2, 2] = -dE1 * xp.exp(-tau / T1)
    return decay, recovery


def evolution_d2_T2(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE2 = tau / T2**3 * (tau / T2 - 2)
    decay[..., 0, 0] *= dE2
    decay[..., 1, 1] *= dE2
    decay[..., 2, 2] = 0
    recovery *= 0
    return decay, recovery


def evolution_d2_g(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dg = (2j * xp.pi * tau) ** 2
    decay[..., 0, 0] *= dg
    decay[..., 1, 1] *= dg
    decay[..., 2, 2] = 0
    recovery *= 0
    return decay, recovery


def evolution_d2_tau_T1(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE1 = (1 - tau / T1) / T1**2
    decay[..., 0, 0] *= 0
    decay[..., 1, 1] *= 0
    decay[..., 2, 2] *= dE1
    recovery[..., 2, 2] = -dE1 * xp.exp(-tau / T1)
    return decay, recovery


def evolution_d2_tau_T2(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE2 = (1 + tau * (-1 / T2 + 2j * xp.pi * g)) / T2**2
    decay[..., 0, 0] *= dE2
    decay[..., 1, 1] *= xp.conj(dE2)
    decay[..., 2, 2] *= 0
    recovery *= 0
    return decay, recovery


def evolution_d2_tau_g(tau, T1, T2, g):
    xp = np
    tau, T1, T2, g = common.expand_arrays(tau, T1, T2, g)
    decay, recovery = _evolution(tau, T1, T2, g)

    dE2 = (1 + tau * (-1 / T2 + 2j * xp.pi * g)) * 2j * xp.pi
    decay[..., 0, 0] *= dE2
    decay[..., 1, 1] *= xp.conj(dE2)
    decay[..., 2, 2] *= 0
    recovery *= 0
    return decay, recovery


# this is 0
# def evolution_d2_T1_T2(tau, T1, T2, g):
# def evolution_d2_T1_g(tau, T1, T2, g):
# def evolution_d2_T2_g(tau, T1, T2, g):
