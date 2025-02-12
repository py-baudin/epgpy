"""Simulate RF-pulse

operator RFPulse:
    Subtype of epg.Sequence.
    Create RF-pulse operator from sequence of complex values,
    duration and RF power / alpha flip angle.

    rf = RFPulse(values, duration, rf/alpha)

utilities:
    estimate_rf:
        estimate RF power (kHz) given target alpha.
    space_to_freq:
        given gradient value (mT/m), convert spatial values (mm) to frequencies (kHz)
    freq_to_space:
        reverse of the above
    encode_phase:
        given gradient value (mT/m) and fov (mm), adds gradient
        dimension to pulse operator

"""

import logging
import numpy as np

from . import functions, operator, opmatrix, statematrix, common, utils
from . import probe, evolution, transition

try:
    from scipy import optimize
except ImportError:
    optimize = None

LOGGER = logging.getLogger(__name__)


class RFPulse(operator.MultiOperator):
    """realistic RF-pulse operator"""

    def __init__(self, values, duration, *, rf=None, alpha=None, phi=None, **kwargs):
        """ Init RF-pulse operator

        The amplitudes and phases of "values" are applied
        respectively as the alpha factors and phi values
        of a sequence of T operators.

        The i-th alpha is computed as follows:
TOFIX        alpha[i] = abs(values[i]) * 2 * Pi * rf * duration / len(values)

        It is often necessary to "rewind" the dephasing at the end

        Parameters:

            values: sequence of complex values
                Values of the discrete pulse

            duration: float
                Total duration of the pulse in ms

            rf: float
                Pulse frequency (in kHz) or attenuation factor
                If not provided, an estimated nominal value will be estimated
                (see alpha below)
                If alpha also provided, this value is multiplied to the nominal rf
                (eg. B1 attenuation)

            alpha: float
                Targeted alpha (in degree)
                Achieved through optimization of "rf"

            phi: float
                Offset angle applied to the pulse (in degree)

            modifier: callable, returns Operator
                function of duration (ms), returning Operator object
                that will be interleaved with pulse operators.
                Use to simulate effect of relaxation, precession, diffusion, etc.

        To simulate the effect of a field gradient on a slice selection:
            * for a given gradient (mT/m),
            * slice_thickness = 30 (mm)
            * nfreq = 100 (number of samples in the profile)
            * gamma = gamma_1H (kHz/T)

            freqs = 1e-6 * gamma * gradient \
                  * slice_thickness * np.linspace(-0.5, 0.5, nfreq)
            pass: g = freqs

        """

        # return  list of operators
        seq, info = rfpulse(values, duration, rf=rf, alpha=alpha, phi=phi, **kwargs)

        # store
        self.values = values
        for item in info:
            setattr(self, item, info[item])

        # create operator
        name = kwargs.pop("name", f"RFPulse({len(values)}, {duration}ms)")
        super().__init__(seq, name=name, duration=duration)


def rfpulse(values, duration, rf=None, alpha=None, phi=None, **kwargs):
    """returns rf-pulse as a sequence of operators"""

    # make complex array
    values = np.asarray(values, dtype=np.complex128)

    if rf is None and alpha is None:
        raise ValueError('Either "rf" or "alpha" must be provided')
    elif rf is None:
        # estimate rf frequency given alpha
        rf = estimate_rf(values, alpha)
    elif alpha is None:
        # estimate alpha given rf
        alpha = estimate_alpha(values, rf)
    # if both alpha and rf, alpha is only stored as reference

    # operator objects
    transform = kwargs.pop("transform", transition.T)

    # make list of operators
    seq = make_pulse_sequence(transform, values, duration, rf, offset=phi)
    info = {"rf": rf, "alpha": alpha, "phi": phi}

    # update sequence with evolution operator
    T1, T2, g = kwargs.get("T1"), kwargs.get("T2"), kwargs.get("g")
    if not all([T1 is None, T2 is None, g is None]):
        # evolution = kwargs.get("evolution", evolution.E)
        T1 = 1e10 if T1 is None else T1
        T2 = 1e10 if T2 is None else T2
        g = 0 if g is None else g
        # seq = functions.modify(seq, evolution, T1=T1, T2=T2, g=g, expand=False)
        seq = functions.modify(seq, T1=T1, T2=T2, g=g, expand=False)
        info.update({"T1": T1, "T2": T2, "g": g})

    return seq, info


def make_pulse_sequence(transform, values, duration, rf, offset=None):
    """return list of Operators from pulse data

    Parameters
    ===
        values: sequence of complex values
            Pulse values
        duration: float
            Pulse duration (ms)
        rf: float
            RF power (kHz)
        offset: float
            phase offset (degree)

    Returns
    ===
        sequence: sequence of Operator objects

    """
    # normalize values
    values = np.asarray(values)
    if values.ndim > 1:
        raise ValueError("`values` array must be 1-dimensional")

    if np.max(np.abs(values)) > 1:
        raise ValueError("pulse values must have a magnitude <= 1")

    nvalue = len(values)

    ndim = len(np.shape(rf))
    if ndim > 1:
        values = values.reshape((nvalue,) + (1,) * ndim)

    # operator durations
    if np.isscalar(duration):
        durations = np.ones(nvalue) * duration / nvalue
    elif len(duration) == nvalue:
        durations = np.asarray(duration)
    else:
        raise ValueError("duration and values must have the same length")

    # series of angles to apply
    # alphas = 360 * np.abs(values) * rf * durations
    alphas = 180 * np.abs(values) * rf
    phis = np.angle(values, deg=True)

    # make list of operators
    sequence = [
        transform(alpha, phi, duration=dur)
        for alpha, phi, dur in zip(alphas, phis, durations)
    ]

    if offset:  # phase offset
        # sequence = [transform(0, -offset)] + sequence + [transform(0, offset)]
        sequence = [transition.Phi(-offset)] + sequence + [transition.Phi(offset)]

    return sequence


def estimate_alpha(values, rf):
    """estimate alpha given RF-amplitude"""
    nvalue = len(values)
    # angles of small rotations
    alphas = rf * 180 * np.abs(values)
    phis = np.angle(values, deg=True)

    # total rotation
    rotation_matrix = opmatrix.matrix_combine_multi(
        transition.rotation_operator(alphas, phis)
    )
    rotation_matrix = common.asarray(rotation_matrix)
    equilibrium = statematrix.StateMatrix([0, 0, 1])

    # apply rotation
    sim = opmatrix.matrix_prod(rotation_matrix, equilibrium.states, inplace=False)

    # longitudinal phase coefficient, rescaled between -1 and +1
    absZ = np.mod(np.real(sim.flat[2]) + 1, 2) - 1

    # resulting alpha angle in degree
    alpha = np.mod(np.arccos(absZ) / np.pi * 180 + 180, 360) - 180
    return alpha


def estimate_rf(values, alpha):
    """Estimate rf value to achieve the target alpha through iterative optimization

    Parameters:
    ===
        values: sequence of complex values
            Pulse values
        alpha: float
            Target flip angle

    Returns
    ===
        rf: float
            RF frequency (kHz) matching for target angle alpha

        TODO: variable step size
    """

    # normalize values
    values = np.asarray(values)
    # nvalue = len(values)

    if np.max(np.abs(values)) > 1:
        raise ValueError("pulse values must have a magnitude <= 1")

    # check if phase constant
    phase_diffs = np.diff(np.mod(np.angle(values, deg=True), 180))
    is_const = np.all(np.isclose(phase_diffs, 0, atol=1e-5))

    if is_const:
        # if pulse has constant phase: no need to optimize
        LOGGER.info(f"Calculate rf for alpha={alpha} (constant phase)")
        # alphas = rf * 180 * np.abs(values)
        rf = alpha / 180 / np.abs(np.sum(values))
        return rf

    #
    # If not constant: need to optimize
    if not optimize:
        raise RuntimeError("Scipy is required for estimating rf")

    LOGGER.info(f"Optimize rf for alpha={alpha}")

    # compute the target state matrix after the pulse
    equilibrium = statematrix.StateMatrix([0, 0, 1])
    ideal_pulse = transition.T(alpha, 90)
    target = common.asnumpy(ideal_pulse(equilibrium).states)

    # alphas = 360 * np.abs(values) * rf * durations
    alphas = 180 * np.abs(values)
    phis = np.angle(values, deg=True)

    def costfunction(rf):
        """cost function"""
        # create operator by combining rotational operators
        rotations = transition.rotation_operator(rf * alphas, phis)
        rotation_matrix = opmatrix.matrix_combine_multi(rotations)

        # apply pulse on initial state matrix
        sim = opmatrix.matrix_prod(rotation_matrix, equilibrium.states, inplace=False)
        sim = common.asnumpy(sim)

        cost = np.sum((np.abs(sim) - np.abs(target)) ** 2)
        return cost

    # def jac(rf):
    #     rots = evolution.rotation(rf * alphas, phis)
    #     sim = bloch.apply(evolution.combine(rots), equilibrium.states, inplace=False)
    #     drot = 0
    #     for i in range(len(alphas)):
    #         drot_i = alphas[i] * evolution.rotation_d_alpha(rf * alphas[i], phis[i])
    #         rots_i = np.concatenate([rots[:i], drot_i, rots[i+1:]], axis=0)
    #         drot = drot + evolution.combine(rots_i)
    #     dsim = bloch.apply(drot, equilibrium.states, inplace=False)
    #     dcost = np.sum(-2 * np.real(dsim.conj() * (target - sim)))
    #     return dcost

    # initial guess
    init = alpha / 180 / np.abs(np.sum(values))

    # run optimization
    result = optimize.minimize(
        costfunction, init, bounds=[(0, None)], tol=1e-8
    )  # , options={'disp': True})

    # optimization result
    LOGGER.info(result)
    rf = result.x[0]

    return rf


#
# profiles and utilities


def encode_phase(
    pulse, gradient, fov, *, expand=True, rewind=None, npoint=101, gamma=utils.gamma_1H
):
    """Modify pulse operator with slice selective gradient (in new dimension)"""
    if not isinstance(pulse, RFPulse):
        raise TypeError("Can only use RFPulse operators")

    if np.isscalar(fov):
        fov = utils.spatial_range(fov, npoint)

    # generate frequency array
    freqs = utils.space_to_freq(gradient, fov, gamma=gamma)
    if expand:
        # make frequency axis as a new axis
        dims = list(range(len(pulse.shape)))
        freqs = common.expand_dims(freqs, dims)

    # modify pulse
    modified = functions.modify(pulse, g=freqs, expand=False)

    if rewind is not None:
        # phase rewinding
        rewind = 0.5 if rewind is True else float(rewind)
        modified.append(evolution.P(pulse.duration * rewind, g=-freqs, duration=0))

    return modified


# def slice_profile(
#     pulse, gradient, fov, *, rewind=True, init=None, npoint=100, profile="F"
# ):
#     """pulse frequency/spacial profile calculation

#     Parameters
#     ===
#         pulse: Pulse Operator
#         gradient: gradient value (mT/m)
#         fov:
#         rewind: [True]/False/float
#             Rewind phase after the pulse with this factor on the gradient integral
#             rewind=True is equivalent to rewind=0.5
#         init: array
#             initial state matrix (see epglib.simulate)

#     Returns
#     ===
#         profile: array of complex values

#     """
#     if np.isscalar(fov):
#         fov = np.linspace(-fov / 2, fov / 2, npoint)

#     # encode phase
#     pulse = encode_phase(pulse, gradient, fov, rewind=rewind)

#     # simulate effect of pulse
#     if profile == "F":
#         adc = probe.ADC
#         return functions.simulate([pulse, adc], init=init)[0]
#     elif profile == "Z":
#         adc = probe.Probe("zsignal")
#         return functions.simulate([pulse, adc], init=init)[0]
#     elif profile == "both":
#         adc = probe.Probe("(signal, zsignal)")
#         return functions.simulate([pulse, adc], init=init)[0]
