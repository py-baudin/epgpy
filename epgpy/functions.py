# Python core imports
import logging

import numpy as np
from . import common, operators, statematrix, utils
from .probe import dft

LOGGER = logging.getLogger(__name__)


def getshape(sequence):
    """extract the overall shape defined by sequence"""
    sequence = flatten_sequence(sequence)
    return common.broadcast_shapes(*[op.shape for op in sequence], append=True)


def getnshift(sequence):
    """determine number of phases states required by sequence"""
    sequence = flatten_sequence(sequence)
    nshift = 0
    for op in sequence:
        nshift += op.nshift
    return nshift


def getkdim(sequence):
    """get number of gradient axes"""
    sequence = flatten_sequence(sequence)
    kdim = 1
    for op in sequence:
        kdim = max(getattr(op, "kdim", 1), kdim)
    return kdim


def get_adc_times(sequence):
    """return ADC opening times (cf. operator's `duration` keyword)"""
    tim = 0
    times = []
    sequence = flatten_sequence(sequence)
    for op in sequence:
        tim = tim + op.duration
        if isinstance(op, operators.Probe):
            times.append(tim)
    return times


def simulate(
    sequence,
    *,
    adc_time=False,
    init=None,
    squeeze=False,
    probe=None,
    callback=None,
    asarray=True,
    disp=False,
    **options,
):
    """simulate a sequence
        Values are returned at operators with the adc flag set to True

    Parameters:
    ===
        sequence: (nested)-list of operators
            Sequence to simulate.
        init: [None], 3-value array
            Initial state (default: equilibrium ie. [0, 0, 1])
        adc_time: True, [False]
            If True, returns the adc opening times
        squeeze: [True], False
            Pre-combine operators when possible
        probe: probe expression (to supersede existing probe)
        callback: callback function after each operator
        asarray: True, [False]
            return output values as single ndarray

        **options: state matrix options (eg. max_nstate, kvalue, kgrid, ...)

     Returns
     ===
        [adc_times]: sequence of float
            Only if adc_time option is True

        values: list of arrays
            Simulated signal at ADC times

    """

    # flatten sub-sequences
    sequence = flatten_sequence(sequence)

    # compute number of gradient shifts
    nshift = getnshift(sequence)
    shape = getshape(sequence)
    LOGGER.info(
        "Simulate sequence: num. operators:"
        f" {len(sequence)}, num. shifts: {nshift}, shape: {shape}"
    )

    # squeeze
    if squeeze:
        LOGGER.info(f"Squeeze sequence")
        sequence = squeeze_sequence(sequence)

    if not any(isinstance(op, operators.Probe) for op in sequence):
        raise ValueError(
            "Cannot simulate sequence without at least one Probe/ADC operator"
        )

    # init state matrix with nshift
    if init is None:
        init = [0, 0, 1]
    else:
        LOGGER.info(f"Simulate: non-default initialization: {init}")

    # custom probe(s)
    probes = []
    if probe:
        LOGGER.info(f"Simulate: custom probe(s): {probe}")
        probes = probe if isinstance(probe, (tuple, list)) else [probe]
        probes = [
            probe if isinstance(probe, operators.Probe) else operators.Probe(probe)
            for probe in probes
        ]

    if callback:
        LOGGER.info(f"Simulate: callback function: {callback}")

    if not isinstance(init, statematrix.StateMatrix):
        # TODO: smart memory pre-allocation (depending on nstate, shape) ?
        nstate = 0  # options.get('max_nstate', 0)
        sm = statematrix.StateMatrix(
            init,
            nstate=nstate,
            shape=shape,
            **options,
        )
    else:
        sm = init
        sm.options.update(options)

    LOGGER.info(
        f"Simulate: initial state matrix:"
        f" num. states: {sm.nstate}"
        f" shape: {sm.shape}"
    )

    # ensure sm is writeable
    sm = sm.copy()

    # display?
    if disp:
        sequence = utils.progressbar(sequence, "Simulating: ")

    tic = 0
    times = []
    values = []
    # for op in sequence:
    for op in sequence:
        # apply each operator in sequence
        sm = op(sm, inplace=True)
        tic = tic + op.duration
        if isinstance(op, operators.Probe):
            # substitute probing operator and store
            values.append(
                [probe.acquire(sm, post=op.post) for probe in (probes or [op])]
            )

            times.append(tic)
        elif callback:
            callback(sm)

    # split multiple measurements
    values = tuple(zip(*values))

    if asarray:
        values = tuple(np.asarray(arr) for arr in values)
        times = np.asarray(times)

    if len(values) == 1:
        # flatten values if only a single acquisiion
        values = values[0]

    # return values
    if adc_time:
        return times, values
    return values


def modify(sequence, modifier=None, *, expand=True, **params):
    """Update sequence with duration-dependant modifier

    Parameters:
        seq: a sequence of operators, some with non-zero duration
        modifier:
            a function whose argument is an operator / an operator whose first argument is a time (eg. T)
            if not set, `operators.E` is implicitely selected, and expected arguments are `T1`, `T2` and `g`
        expand: [True]/False
            If True, non-scalar parameters are added as new dimensions to the sequence.
            If False, ovelaping dimensions of the parameters and sequence must match.
        params: modifier's parameters (eg. T1, T2, g, ...)

    Example:
        # spin echo
        seq = [T(90, 90), S(1, duration=1), T(180, 0), S(1, duration=1), ADC]
        seq2 = modify(seq, T1=100, T2=30)
        > seq2
        > [T(90, 90), S(1) | E(1, 100, 30), T(180, 0), S(1) | E(1, 100, 30), ADC]

    """

    shape = getshape(sequence)
    values = common.expand_arrays(*params.values(), append=True)
    if expand and (len(shape) > 1 or shape[0] > 1):
        dims = range(len(shape))
        values = common.map_arrays(values, lambda arr: common.expand_dims(arr, dims))
    params = dict(zip(params, values))

    if not modifier:
        # default modifier handles T1, T2, g and 'att' (B1-attenuation)
        modifier = default_modifier
        if not params:
            LOGGER.info(f"Modify sequence: nothing to do")
            return sequence
    elif not callable(modifier):
        raise TypeError(f"`modifier` must be a callable")

    newseq = []
    opdict = {}

    for op in flatten_sequence(sequence):
        if op in opdict:
            # already modified
            newseq.append(opdict[op])
            continue

        # else: build new operator
        op0 = op

        # apply modifier
        op = modifier(op, **params)

        # store modified operator
        opdict[op0] = op
        newseq.append(op)

    # verify shape
    LOGGER.info(f"Modify sequence: {shape}->{getshape(newseq)}")

    if isinstance(sequence, operators.MultiOperator):
        return operators.MultiOperator(newseq, name=sequence.name)
    return newseq


def default_modifier(op, **kwargs):
    """default modifier to handle 'T1', 'T2', 'g' and 'att' keywords
    TODO: handle differential operators (options gradients and hessian)
    """
    if isinstance(op, operators.T):
        # add B1 attenuation
        att = kwargs.get("att")
        if att is None or np.allclose(att, 1):
            pass  # nothing to do
        else:
            # update T operator
            op = operators.T(op.alpha * att, op.phi, name=op.name, duration=op.duration)
            op.name += "#"

    if np.any(op.duration > 0):
        # add relaxation or precession
        T1, T2, g = kwargs.get("T1"), kwargs.get("T2"), kwargs.get("g")
        if T1 is None and T2 is None and g is None:
            pass  # nothing to do
        elif T1 is None and T2 is None:
            # precession only
            op = op * operators.P(op.duration, g, duration=0)
            op.name = op[0].name + "*"
        else:
            # relaxation
            T1 = 1e10 if T1 is None else T1
            T2 = 1e10 if T2 is None else T2
            g = 0 if g is None else g
            op = op * operators.E(op.duration, T1, T2, g, duration=0)
            op.name = op[0].name + "*"
    # return modified operator
    return op


def squeeze_sequence(seq):
    """merge repeated sequences of operators for speed"""
    raise NotImplementedError("Automatic sequence squeezing not implemented yet")


def flatten_sequence(seq, flatten_multi=True):
    """return a flat list of operators"""
    seq = [seq] if isinstance(seq, operators.Operator) else seq

    newseq = []
    for item in seq:
        if isinstance(item, list):
            newseq.extend(flatten_sequence(item))
        elif flatten_multi and isinstance(item, operators.MultiOperator):
            newseq.extend(flatten_sequence(item.operators))
        elif isinstance(item, operators.Operator):
            newseq.append(item)
        else:
            raise ValueError(f"Invalid operator: {item}")
    return newseq
