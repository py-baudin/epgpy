""" unittest for epgpy.sequence """

import numpy as np
import pytest
from epgpy import sequence, core, diff
from epgpy.sequence import Function, Variable, Constant, Expression, math, Sequence


def test_expression_class():
    a = Variable("a")
    b = Variable("b")
    const = Constant(1)

    assert a(a=1) == 1
    assert const() == 1

    # single operator
    assert (a + 2)(a=1) == 3
    assert (2 - a)(a=1) == 1
    assert (a * 2)(a=1) == 2
    assert (1 / a)(a=2) == 0.5
    assert (a**2.0)(a=2) == 4
    assert (a**0.5)(a=4) == 2
    assert (2**a)(a=3) == 8

    # multiple operators
    assert (a * 2 + 1)(a=1) == 3
    assert ((1 / a) ** 2)(a=2) == 1 / 4

    # multiple variables
    assert (a + b)(a=1, b=2) == 3
    assert (a - b)(a=1, b=2) == -1
    assert (a * b)(a=1, b=2) == 2
    assert (a / b)(a=1, b=2) == 0.5
    assert (a**b)(a=2, b=2) == 4

    # complicated
    assert (2 * a + 1 / b)(a=0.25, b=2) == 1
    assert ((1 / a) ** (1 / b))(a=0.25, b=2) == 2

    # const
    assert (const + 2)() == 3
    assert (const + 2).derive("a") == 0

    # derivatives
    assert const.derive("a") == 0
    assert (a + 1).derive("a", a=2) == 1
    assert (a * 3).derive("a", a=2) == 3
    assert (1 - a).derive("a", a=2) == -1
    assert (1 / a).derive("a", a=2) == -0.25
    assert (a**2).derive("a", a=3) == 6
    assert (2**a).derive("a", a=2) == np.log(2) * 4

    # multiple variables
    assert (a - b).derive("a", a=1, b=1) == 1
    assert (a - b).derive("b", a=1, b=1) == -1
    assert (a * b).derive("a", a=1, b=2) == 2
    assert (a * b).derive("b", a=1, b=2) == 1
    assert (a / b).derive("a", a=1, b=2) == 0.5
    assert (a / b).derive("b", a=1, b=2) == -0.25
    assert (a**b).derive("a", a=3, b=2) == 6
    assert (a**b).derive("b", a=3, b=2) == np.log(3) * 9

    # complicated
    # deriv_a = 2*a - 2/(2 * a - b)**2
    assert (a**2 + 1 / (2 * a - b)).derive("a", a=1, b=1) == (2 - 2 / (2 - 1) ** 2)
    # deriv_b =  1/(2* a - b)
    assert (a**2 + 1 / (2 * a - b)).derive("b", a=1, b=1) == 1 / (2 - 1) ** 2

    #  math
    math = sequence.math
    assert math.log(a)(a=2) == np.log(2)
    assert math.exp(a)(a=2) == np.exp(2)
    assert math.sqrt(a)(a=2) == np.sqrt(2)
    assert math.abs(a)(a=-2) == np.abs(-2)
    assert math.cos(a)(a=np.pi / 3) == np.cos(np.pi / 3)
    assert math.sin(a)(a=np.pi / 3) == np.sin(np.pi / 3)
    assert math.tan(a)(a=np.pi / 3) == np.tan(np.pi / 3)

    assert math.log(a).derive("a", a=2) == 0.5
    assert math.exp(a).derive("a", a=2) == np.exp(2)
    assert math.sqrt(a).derive("a", a=2) == 0.5 / np.sqrt(2)
    assert math.abs(a).derive("a", a=-2) == -1
    assert math.cos(a).derive("a", a=np.pi / 3) == -np.sin(np.pi / 3)
    assert math.sin(a).derive("a", a=np.pi / 3) == np.cos(np.pi / 3)
    assert math.tan(a).derive("a", a=np.pi / 3) == 1 / np.cos(np.pi / 3) ** 2

    # combined
    assert math.log(1 + math.exp(2 * a))(a=1) == np.log(1 + np.exp(2))
    assert math.cos(a + math.sin(b / 2))(a=1, b=np.pi) == np.cos(1 + np.sin(np.pi / 2))
    assert (a * math.log(b)).derive("a", a=2, b=3) == np.log(3)
    assert (a * math.log(b)).derive("b", a=2, b=3) == 2 / 3

    # variables
    assert (2 + const).variables == set()
    assert a.variables == {"a"}
    assert (a + 2 * b + math.log(a)).variables == {"a", "b"}

    # error catching
    with pytest.raises(TypeError):
        Constant("a")  # constant must be a numeric type
    with pytest.raises(TypeError):
        Constant(None)

    # n-dimensional
    variable = Variable("a")
    constant = Constant([[1, 2, 3], [4, 5, 6]])

    # constant
    assert np.allclose((2 * constant)(), [[2, 4, 6], [8, 10, 12]])
    # variable
    assert np.allclose(2 * variable(a=[1, 2, 3]), [2, 4, 6])
    assert np.allclose((variable + constant)(a=2), [[3, 4, 5], [6, 7, 8]])
    assert np.allclose(math.sqrt(variable + 1)(a=[1, 2, 3]), np.sqrt([2, 3, 4]))
    # broadcasting
    assert np.allclose((variable + constant)(a=[1, 2, 3]), [[2, 4, 6], [5, 7, 9]])

    # equality
    assert Variable("a") == Variable("a")
    assert Variable("a") != Variable("b")
    assert const == Constant(1)
    assert const != Constant(2)
    assert const != a
    assert (Variable("a") + 1) != Variable("a")


def test_function_class():
    log = Function(np.log, derivative=np.reciprocal)
    assert isinstance(log(1), Expression)
    assert isinstance(log("a"), Expression)
    assert np.isclose(log(1)(), 0.0)
    assert np.isclose(log("a")(a=1), 0.0)

    custom = Function(
        np.divide, derivative=[lambda x, y: 1 / y, lambda x, y: -x / y**2]
    )
    assert isinstance(custom("a", "b"), Expression)
    assert custom("a", "b").variables == {"a", "b"}
    assert np.isclose(custom("a", "b")(a=1, b=2), 0.5)
    assert np.isclose(custom("a", "b").derive("a", a=1, b=2), 0.5)
    assert np.isclose(custom("a", "b").derive("b", a=1, b=2), -1 / 2**2)
    # assert np.allclose(custom("a", "b").gradient(a=1, b=2), [0.5, -1 / 2 ** 2])

    custom2 = Function(np.sum, axis=1)
    sum1 = custom2("a")  # pass options
    arr = np.arange(2 * 3).reshape(2, 3)
    assert np.allclose(sum1(a=arr), arr.sum(axis=1))


def test_operator_class():
    ops = sequence.operators
    sm0 = core.StateMatrix()

    # constant
    op = ops.T(90, 90)
    assert op.variables == set()
    assert isinstance(op.build(), core.Operator)
    assert np.allclose(op.build()(sm0), core.T(90, 90)(sm0))

    # variable
    alpha = Variable("att") * 90
    op = ops.T(alpha, 90)
    assert op.variables == {"att"}
    assert op.build(att=0.9).alpha == 0.9 * 90
    assert np.allclose(op.build(att=0.9)(sm0), core.T(0.9 * 90, 90)(sm0))

    with pytest.raises(ValueError):
        op.build()  # missing att value

    # gradient
    sm1 = op.build(gradient="att", att=0.9)(sm0)
    assert "att" in sm1.gradient
    # finite differences
    fdiff = (core.T(90 * (0.9 + 1e-8), 90)(sm0).states - sm1.states) * 1e8
    assert np.allclose(fdiff, sm1.gradient["att"])

    # multiple variables
    op = ops.E(10, "T1", "T2")
    assert op.variables == {"T1", "T2"}
    sm0 = core.StateMatrix([1, 1, 0])
    sm1 = op.build(T1=1000, T2=35)(sm0)
    assert np.allclose(sm1, core.E(10, 1000, 35)(sm0))
    with pytest.raises(ValueError):
        op.build(T1=1000)  # missing T2
    with pytest.raises(ValueError):
        op.build(T2=35)  # missing T1
    sm1 = op.build(gradient="T2", T1=1000, T2=35)(sm0)
    fdiff = (core.E(10, 1000, 35 + 1e-8)(sm0).states - sm1.states) * 1e8
    assert np.allclose(fdiff, sm1.gradient["T2"])

    # fix some variables
    relax = ops.E("tau", "T1", "T2", "g")
    assert relax.variables == {"tau", "T1", "T2", "g"}
    relax2 = relax(tau=10)
    assert relax2.variables == {"T1", "T2", "g"}
    assert isinstance(relax2._expressions["tau"], Constant)
    assert relax2._expressions["tau"].value == 10

    relax3 = relax2(T1=1000, T2=30, g=0)
    assert relax3.variables == set()

    # equality
    assert ops.E("tau", "T1", 30) == ops.E("tau", "T1", 30)
    assert ops.E("tau", "T1", 30) != ops.E("tau", 1000, 30)
    assert ops.E("tau", "T1", 30) != ops.E("tau", "T1", "T2")
    assert ops.E("tau", "T1", 30) != ops.E("tau", "T1", 35)
    assert ops.E("tau", "T1", 30, name="foo") == ops.E("tau", "T1", 30, name="foo")
    assert ops.E("tau", "T1", 30, name="foo") != ops.E("tau", "T1", 30, name="bar")

    # Adc
    adc = ops.Adc(phase=None)
    assert adc.build().phase is None
    adc = ops.Adc(phase=15)
    assert adc.build().phase == 15
    adc = ops.Adc(phase="phi")
    assert adc.build(phi=10).phase == 10

    adc = ops.Adc(weights="weights", reduce=1)
    assert np.allclose(adc.build(weights=[[1, 2]]).weights, [[1, 2]])
    assert adc.build(weights=[[1, 2]]).reduce == (1,)


def test_sequence_class():
    ops = sequence.operators

    T2 = Variable("T2")
    excit = ops.T(90, 90)
    refoc = ops.T(180, 0)
    shift = ops.S(1, duration=5)
    relax = ops.E(5, 1400, T2)
    necho = 5
    adc = ops.ADC

    # sequence object
    seq = Sequence([excit] + [shift, relax, refoc, shift, relax, adc] * necho)
    assert seq.variables == {"T2"}
    assert len(seq) == necho * 6 + 1
    assert list(seq) == seq.operators
    assert seq[-1] is seq.operators[-1]

    # build sequence
    oplist = seq.build(T2=35)
    assert all(isinstance(op, core.Operator) for op in oplist)
    assert oplist[0].alpha == 90
    assert oplist[0].phi == 90
    assert oplist[1].k == 1
    assert oplist[2].tau == 5
    assert oplist[2].T1 == 1400
    assert oplist[2].T2 == 35  # variable parameter

    # make sure operators are not duplicated
    assert oplist[1] is oplist[4]
    assert oplist[2] is oplist[5]
    assert oplist[1] is oplist[7]

    # adc times
    times = seq.adc_times(T2=35)
    assert np.allclose(times, (np.arange(5) + 1) * 10)

    # simulate sequence
    signal = seq.simulate(T2=35)
    assert np.allclose(core.simulate(oplist), signal)

    # derive
    grad = seq.derive("T2", T2=35)
    # finite differences
    assert np.allclose(1e8 * (seq.simulate(T2=35 + 1e-8) - signal), grad)

    # simulate options
    seq = Sequence([excit] + [shift, relax, refoc, shift, relax, adc] * necho)
    mats = seq.simulate(probe="states", asarray=False, T2=35)
    assert mats[-1].shape == (1, 10 * 2 + 1, 3)

    seq = Sequence(
        [excit] + [shift, relax, refoc, shift, relax, adc] * necho, max_nstate=5
    )
    mats = seq.simulate(probe="states", asarray=False, T2=35)
    assert mats[-1].shape == (1, 5 * 2 + 1, 3)

    # string operators
    seq = Sequence([excit, relax, "SPOILER", "ADC"])
    assert isinstance(seq.build(T2=35)[2], core.Spoiler)
    assert isinstance(seq.build(T2=35)[3], core.Probe)

    # sequence of sequence
    spinecho = Sequence([shift, relax, refoc, shift, relax, adc], max_nstate=5)
    seq = Sequence([excit] + [spinecho] * necho)
    assert len(seq) == necho + 1
    assert len(seq.flatten()) == 1 + necho * len(spinecho)
    assert seq.variables == spinecho.variables  # variables are passed
    assert seq.options["max_nstate"] == 5  # options are passed to the outer sequence


def test_sequence_multiple_variables():
    ops = sequence.operators

    excit = ops.T(90, 90)
    refoc = ops.T("alpha", 0)
    shift = ops.S(1, duration=5)
    relax = ops.E(5, "T1", "T2")
    necho = 5
    adc = ops.ADC

    # sequence object
    seq = Sequence([excit] + [shift, relax, refoc, relax, shift, adc] * necho)
    assert seq.variables == {"T2", "T1", "alpha"}

    # measure signal and derivative
    signal = seq.signal(T2=35, T1=1000, alpha=150)
    assert signal.shape == (1, necho)

    dT2 = seq.derive("T2", T2=35, T1=1000, alpha=150)[..., 0]
    dT1 = seq.derive("T1", T2=35, T1=1000, alpha=150)[..., 0]
    dalpha = seq.derive("alpha", T2=35, T1=1000, alpha=150)[..., 0]

    # gradient
    _, grad = seq.gradient(["alpha", "T1", "T2"], T2=35, T1=1000, alpha=150)
    assert grad.shape == (1, necho, 3)
    assert np.allclose(grad, np.stack([dalpha, dT1, dT2], axis=1))

    fdiff = (
        np.stack(
            [
                seq.signal(T2=35, T1=1000, alpha=150 + 1e-8),
                seq.signal(T2=35, T1=1000 + 1e-8, alpha=150),
                seq.signal(T2=35 + 1e-8, T1=1000, alpha=150),
            ],
            axis=2,
        )
        - signal[..., np.newaxis]
    )
    assert np.allclose(fdiff * 1e8, grad, atol=1e-7)

    # hessian
    _, _, hess = seq.hessian(["alpha", "T1", "T2"], T2=35, T1=1000, alpha=150)
    assert hess.shape == (1, necho, 3, 3)

    fdiff = (
        np.stack(
            [
                np.stack(
                    [
                        seq.derive("alpha", T2=35, T1=1000, alpha=150 + 1e-8)[..., 0],
                        seq.derive("alpha", T2=35, T1=1000 + 1e-8, alpha=150)[..., 0],
                        seq.derive("alpha", T2=35 + 1e-8, T1=1000, alpha=150)[..., 0],
                    ],
                    axis=1,
                ),
                np.stack(
                    [
                        seq.derive("T1", T2=35, T1=1000, alpha=150 + 1e-8)[..., 0],
                        seq.derive("T1", T2=35, T1=1000 + 1e-8, alpha=150)[..., 0],
                        seq.derive("T1", T2=35 + 1e-8, T1=1000, alpha=150)[..., 0],
                    ],
                    axis=1,
                ),
                np.stack(
                    [
                        seq.derive("T2", T2=35, T1=1000, alpha=150 + 1e-8)[..., 0],
                        seq.derive("T2", T2=35, T1=1000 + 1e-8, alpha=150)[..., 0],
                        seq.derive("T2", T2=35 + 1e-8, T1=1000, alpha=150)[..., 0],
                    ],
                    axis=1,
                ),
            ],
            axis=2,
        )
        * 1e8
        - grad[..., np.newaxis, :] * 1e8
    )

    assert np.allclose(fdiff, hess, atol=1e-7)

    # include magnitude as variable
    _, grad, hess = seq.hessian(["magnitude", "alpha"], T2=35, T1=1000, alpha=150)
    fdiff = (
        np.stack(
            [
                seq.signal(T2=35, T1=1000, alpha=150) * (1 + 1e-8) - signal,
                seq.signal(T2=35, T1=1000, alpha=150 + 1e-8) - signal,
            ],
            axis=-1,
        )
        * 1e8
    )
    assert np.allclose(grad, fdiff, atol=1e-7)

    fdiff = (
        np.stack(
            [
                np.concatenate(
                    [
                        seq.gradient("magnitude", T2=35, T1=1000, alpha=150)[1],
                        seq.gradient("magnitude", T2=35, T1=1000, alpha=150 + 1e-8)[1],
                    ],
                    axis=-1,
                ),
                np.concatenate(
                    [
                        seq.gradient("alpha", T2=35, T1=1000, alpha=150)[1]
                        * (1 + 1e-8),
                        seq.gradient("alpha", T2=35, T1=1000, alpha=150 + 1e-8)[1],
                    ],
                    axis=-1,
                ),
            ],
            axis=-1,
        )
        * 1e8
    )
    fdiff = fdiff - grad[..., np.newaxis, :] * 1e8
    assert np.allclose(hess, fdiff, atol=1e-7)

    # CRLB
    variables = ["alpha", "T2"]
    crlb, d_crlb = seq.crlb(variables, T2=35, T1=1000, alpha=150, gradient=True)
    fdiff = (
        np.stack(
            [
                seq.crlb(variables, T2=35, T1=1000, alpha=150 + 1e-8) - crlb,
                seq.crlb(variables, T2=35 + 1e-8, T1=1000, alpha=150) - crlb,
            ],
            axis=-1,
        )
        * 1e8
    )
    assert np.allclose(fdiff, d_crlb, rtol=1e-4)


def test_partial_hessian():
    ops = sequence.operators

    excit = ops.T(90, 90)
    refoc = ops.T("alpha", 0)
    shift = ops.S(1, duration=5)
    relax = ops.E(5, "T1", "T2")
    necho = 5
    adc = ops.ADC

    # sequence object
    seq = Sequence([excit] + [shift, relax, refoc, shift, relax, adc] * necho)

    # 2nd derivative for all 9 pairs
    sig1, grad1, hess1 = seq.hessian(["alpha", "T1", "T2"], alpha=150, T1=1e3, T2=30)
    assert hess1.shape == (1, 5, 3, 3)
    # 2nd derivatives for 2 selected pairs
    sig2, grad2, hess2 = seq.hessian(["T1", "T2"], ["alpha"], alpha=150, T1=1e3, T2=30)
    assert hess2.shape == (1, 5, 2, 1)
    # alpha-T1
    assert np.allclose(hess1[..., 0, 1], hess2[..., 0, 0])
    assert np.allclose(hess1[..., 1, 0], hess2[..., 0, 0])
    assert np.allclose(hess1[..., 0, 2], hess2[..., 1, 0])
    assert np.allclose(hess1[..., 2, 0], hess2[..., 1, 0])

    # crlb
    crlb1, cgrad1 = seq.crlb(["T1", "T2"], gradient=["alpha"], alpha=150, T1=1e3, T2=30)
    assert len(grad1) == 1

    crlb2, cgrad2 = sequence.optim.crlb(grad2, H=hess2)
    assert np.allclose(crlb1, crlb2)
    assert np.allclose(cgrad1, cgrad2)


def test_rfpulse():
    operators = sequence.operators

    values = np.sin(np.linspace(0, 1, 10) * np.pi)
    rf = operators.RFPulse(values, "duration", rf="rf")

    _rf = rf.build(duration=5, rf=1)
    assert isinstance(_rf, core.Operator)
    assert np.allclose(values, _rf.values)
    assert np.allclose(5, _rf.duration)
    assert np.allclose(1, _rf.rf)

    seq = Sequence([rf, operators.ADC])
    sig = seq.signal(duration=5, rf=1)
    assert np.allclose(sig, _rf(core.StateMatrix()).F0)

    # with decay/precession
    rf = operators.RFPulse(values, duration=5, rf=1, T2="T2", g="g")
    _rf = rf.build(T2=30, g=-0.1)
    assert len(_rf) == 20
    assert isinstance(_rf[1], core.E)
