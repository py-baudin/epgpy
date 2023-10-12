import pytest
import numpy as np
from epgpy import statematrix, functions, operators, utils


def test_functions():
    """ """
    excit = operators.T(90, 90)
    refoc = operators.T(180, 0)
    grad = operators.S(1, duration=10)
    relax = operators.E(10, 1000, 30)
    adc = operators.ADC

    seq1 = [excit, grad, relax, refoc, grad, relax, adc]
    seq2 = excit * grad * relax * refoc * grad * relax * adc

    assert seq1[0] == seq2[0] is excit
    assert seq1[1] == seq2[1] is grad
    assert seq1[2] == seq2[2] is relax
    assert seq1[3] == seq2[3] is refoc
    assert seq1[4] == seq2[4] is grad
    assert seq1[5] == seq2[5] is relax
    assert seq1[6] == seq2[6] is adc

    # getnshift
    assert functions.getnshift(seq1) == functions.getnshift(seq2) == seq2.nshift == 2

    # get shape
    assert functions.getshape(seq1) == functions.getshape(seq2) == (1,)

    # get adc times
    assert functions.get_adc_times(seq1) == functions.get_adc_times(seq2) == [20]

    # simulate
    signal_1 = functions.simulate(seq1)
    signal_2 = functions.simulate(seq2)
    assert np.allclose(signal_1, signal_2)

    with pytest.raises(ValueError):
        functions.simulate([operators.T(90, 90)])  # no operators.ADC

    # custom probe
    seq3 = list(seq1)
    seq3[-1] = operators.Probe("(real(F0), imag(F0))")
    assert functions.get_adc_times(seq3) == [20]
    res = functions.simulate(seq3)
    assert np.allclose(res[0], [np.real(signal_1[0]), np.imag(signal_1[0])])

    # if passed, probe argument is prevalent
    res = functions.simulate(seq3, probe="abs(F0)")
    assert np.allclose(res, np.abs(signal_1))

    # multiple probes
    signal_f0 = functions.simulate(seq3, probe="F0")
    signal_z0 = functions.simulate(seq3, probe="Z0")
    res = functions.simulate(seq3, probe=["F0", "Z0"])
    assert np.allclose(res[0], signal_f0)
    assert np.allclose(res[1], signal_z0)

    # phase compensation
    adc2 = operators.Adc(phase=15)
    seq4 = [excit, grad, relax, refoc, grad, relax, adc2]
    res4 = functions.simulate(seq4)
    assert np.isclose(res4, signal_f0 * np.exp(1j * 15 / 180 * np.pi))
    # compensation also work if probe operator is replaced
    res4 = functions.simulate(seq4, probe="Z0")
    assert np.isclose(res4, signal_z0 * np.exp(1j * 15 / 180 * np.pi))

    # weights and reduce
    adcn = operators.Adc(reduce=1, weights=[[1, 2, 3, 4, 5]])
    relaxn = relax = operators.E(
        10, 1000, [[30], [40], [50]], g=[[-0.1, -0.05, 0, 0.05, 1]]
    )
    res_ = functions.simulate([excit, grad, relaxn, refoc, grad, relaxn, adc])
    resn = functions.simulate([excit, grad, relaxn, refoc, grad, relaxn, adcn])
    assert np.allclose(np.dot(res_, [1, 2, 3, 4, 5]), resn)


def test_simulate_ndim():
    ax = utils.Axes("FA", "T2")

    excit = operators.T(90, 90)
    refoc = operators.T([180, 150], 0, axes=ax.FA)
    relax = operators.E(10, 1e3, [30, 40, 50], axes=ax.T2)
    grad = operators.S(1)
    adc = operators.ADC
    necho = 2
    seq = [excit] + [grad, relax, refoc, grad, relax, adc] * necho

    # check shape, axes, nshift
    assert functions.getshape(seq) == (2, 3)
    assert functions.getnshift(seq) == 4

    signal = functions.simulate(seq)
    assert all(sig.shape == (2, 3) for sig in signal)

    # custom init
    signal = functions.simulate(seq, init=statematrix.StateMatrix(shape=(1, 1, 4)))
    assert all(sig.shape == (2, 3, 4) for sig in signal)

    with pytest.raises(ValueError):
        # incompatible shape
        functions.simulate(seq + [operators.T([90] * 3, 180)])

    with pytest.raises(ValueError):
        # incompatible shape
        functions.simulate(seq, init=statematrix.StateMatrix(shape=(3, 3)))


def test_modify():
    """test modify function"""
    pulse = operators.T(90, 0, duration=1)
    grad = operators.S(1, duration=5)
    seq = [pulse, grad, pulse, operators.ADC]

    assert seq[0] is seq[2]  # only one pulse operator

    # modify with dummy modifiers
    newseq = functions.modify(seq, lambda op: op)
    assert newseq == seq

    # add some modifiers
    newseq = functions.modify(seq, T2=100)
    assert len(newseq) == len(seq)
    assert newseq[0] is newseq[2]  # only one pulse operator

    # check sequence timing is not modified
    assert functions.get_adc_times(seq) == functions.get_adc_times(newseq)
    assert all(op1.duration == op2.duration for op1, op2 in zip(seq, newseq))

    # check flattened sequence operators
    flatseq = functions.flatten_sequence(newseq)
    assert isinstance(flatseq[0], operators.T)
    assert flatseq[0].alpha == 90
    assert flatseq[0].duration == seq[0].duration  # time unchanged

    assert isinstance(flatseq[1], operators.E)
    assert flatseq[1].T2 == 100  # T2 is set
    assert flatseq[1].duration == 0  # no time added

    # test custom modifier
    modseq = functions.flatten_sequence(functions.modify(seq, T2=30))
    assert len(modseq) == 7
    assert modseq[1].tau == modseq[0].duration
    assert modseq[3].tau == modseq[2].duration
    assert modseq[5].tau == modseq[4].duration

    # multi-dim
    seq = [operators.T(90, 90), operators.Wait(1), operators.T(90, 90), operators.ADC]

    newseq = functions.modify(seq, g=[[0, 0.25, 0.5]], att=[1, 0.5])
    assert functions.getshape(newseq) == (2, 3)
    signal = functions.simulate(newseq)[0]
    assert signal.shape == (2, 3)

    assert np.isclose(signal[0, 0], 0)  # full inversion
    assert np.isclose(signal[0, 1], 1j)  # saturation on y
    assert np.isclose(signal[0, 2], 0)

    assert np.isclose(signal[1, 0], 1)  # saturation on x
    assert np.isclose(
        signal[1, 1],
        functions.simulate([operators.T(45, 180), operators.T(45, 90), operators.ADC]),
    )

    # chain modify
    seq1 = [operators.T(90, 90), operators.Wait(1), operators.T(90, 90), operators.ADC]
    assert functions.getshape(seq1) == (1,)

    seq2 = functions.modify(seq1, T2=[30, 40])
    assert functions.getshape(seq2) == (2,)

    seq3 = functions.modify(seq2, att=[1, 0.9, 0.7])
    assert functions.getshape(seq3) == (2, 3)

    # expand option set to False
    seq4 = functions.modify(seq2, T2=[50, 60], expand=False)
    assert functions.getshape(seq4) == (2,)

    with pytest.raises(ValueError):
        # invalid shape
        functions.modify(seq2, att=[1, 0.9, 0.7], expand=False)

    # test custom modifier
    def modifier(op, x):
        if not isinstance(op, operators.T):
            return op
        return operators.T(op.alpha, op.phi * np.asarray(x))

    seqc = functions.modify(seq1, modifier, x=0.1)
    assert np.allclose(seqc[0].phi, seq[0].phi * 0.1)
    seqc = functions.modify(seq1, modifier, x=[0.1, 0.2])
    assert np.allclose(seqc[0].phi, seq[0].phi * np.r_[0.1, 0.2])


# @pytest.mark.skip(reason="squeeze not implemented")
# def test_squeeze_sequence():
#     """test squeeze function"""
#     seq1 = [
#         operators.T(45, 90),
#         operators.E(1, 100, 30, g=0.25),
#         operators.T(45, 180),
#         operators.E(1, 100, 30),
#         operators.ADC,
#     ]
#     seq2 = functions.squeeze_sequence(seq1)
#     assert len(seq2) == 2

#     # compare results
#     sm = statematrix.StateMatrix()
#     seq1 = MultiOperator(seq1)
#     seq2 = MultiOperator(seq2)
#     assert np.allclose(seq1(sm).states, seq2(sm).states)

#     # ndim
#     T1 = [100, 1e8]
#     seq3 = [
#         operators.T(45, 90),
#         operators.E(1, T1, 30, g=0.25),
#         operators.T(45, 180),
#         operators.E(1, T1, 30),
#         operators.ADC,
#     ]
#     seq4 = functions.squeeze_sequence(seq3)

#     seq3 = MultiOperator(seq3)
#     seq4 = MultiOperator(seq4)
#     assert np.allclose(seq3(sm).states, seq4(sm).states)
#     assert np.allclose(seq1(sm).states, seq4(sm).states[0])

#     # with phase states
#     seq5 = [
#         operators.T(45, 90),
#         operators.E(1, T1, 30, g=0.25),
#         operators.S(1),
#         operators.T(45, 180),
#         operators.E(1, T1, 30),
#         operators.S(1),
#         operators.ADC,
#     ]
#     seq6 = functions.squeeze_sequence(seq5)
#     seq5 = MultiOperator(seq5)
#     seq6 = MultiOperator(seq6)
#     assert len(seq6) == 5
#     assert np.allclose(seq5(sm).states, seq6(sm).states)
