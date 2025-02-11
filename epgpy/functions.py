# Python core imports
import os
import signal
import logging
import multiprocessing as mp
import numpy as np
from . import common, operators, statematrix, utils
from .utils import dft, imaging

LOGGER = logging.getLogger(__name__)
Probe = operators.Probe


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
        probe: probe expression to supersede existing probe (or list of).
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
        LOGGER.info(f"Non-default initialization: {init}")

    # custom probe(s)
    probes = []
    if probe:
        LOGGER.info(f"Custom probe(s): {probe}")
        probes = probe if isinstance(probe, (tuple, list)) else [probe]
        # list probe operators (if None, use operator in sequence)
        probes = [
            probe if isinstance(probe, (Probe, type(None))) else Probe(probe)
            for probe in probes
        ]

    if callback:
        LOGGER.info(f"Callback function: {callback}")

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

    LOGGER.info(f"Initial state matrix: num. states: {sm.nstate}, shape: {sm.shape}")

    # ensure sm is writeable
    sm = sm.copy()

    # run simulation
    values, times = simulate_simple(
        sm, sequence, probes=probes, callback=callback, disp=disp
    )

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


def simulate_simple(sm, sequence, probes=None, callback=None, disp=False):
    """simple sequence loop"""
    if disp:  # display?
        sequence = utils.progressbar(sequence, "Simulating: ")

    tic = 0
    times, values = [], []
    for op in sequence:
        # apply each operator in sequence
        sm = op(sm, inplace=True)
        tic = tic + op.duration
        if isinstance(op, Probe):
            # substitute probing operator and store (None is replaced with op)
            values.append(
                [(pb or op).acquire(sm, post=op.post) for pb in (probes or [op])]
            )
            times.append(tic)
        elif callback:
            callback(sm)
    return values, times


# def mp_initializer():
#     """Ignore CTRL+C in the worker process."""
#     signal.signal(signal.SIGINT, signal.SIG_IGN)


# def simulate_parallel(
#     sm, sequence, probes=None, callback=None, split="order2", njob=None, disp=False
# ):
#     """parallel sequence simulation (experimental)"""
#     njob = njob or (os.cpu_count() - 1)
#     seqs = []
#     if split == "order2":  # split on order2
#         LOGGER.info(f"order-2 parallel simulate with n.jobs: {njob}")
#         order1 = {param for op in sequence for param in getattr(op, "order1", {})}
#         order2 = sorted({pair for op in sequence for pair in getattr(op, "order2", {})})
#         for i in range(njob):
#             order2_i = set(order2[i::njob])
#             order1_i = order1 & {param for pair in order2_i for param in pair}
#             seq_i = []
#             for op in sequence:
#                 if not (getattr(op, "order1", None) or getattr(op, "order2", None)):
#                     seq_i.append(op)
#                     continue
#                 op2 = op.copy()
#                 op2.order1 = {
#                     param: op2.order1[param] for param in set(op2.order1) & order1_i
#                 }
#                 op2.order2 = set(op2.order2) & order2_i
#                 seq_i.append(op2)
#             seqs.append(seq_i)
#     else:
#         raise ValueError(f"Unknown split argument: {split}")

#     # parallel simulate
#     args = [(sm, seqs[i], probes, None, False) for i in range(njob)]
#     with mp.Pool(njob, initializer=mp_initializer) as pool:
#         try:
#             values = pool.starmap(simulate_simple, args)
#         except KeyboardInterrupt:
#             pool.terminate()
#             pool.join()
#             raise KeyboardInterrupt()

#     # merge values
#     mp_values, times = zip(*values)
#     times = times[0]
#     values = mp_values[0]
#     for i in range(1, njob):
#         nadc = len(values)
#         for n in range(nadc):
#             nprobe = len(values[n])
#             for p in range(nprobe):
#                 values[n][p] = values[n][p] + mp_values[i][n][p]
#     return values, times


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
