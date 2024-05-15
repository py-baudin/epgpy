""" unittest for epgpy.diff """
import pytest
import numpy as np
from epgpy import core, diff, common


def test_parse_partials():
    class Op:
        parameters = ["x", "y"]

        def __repr__(self):
            return "op"

    op = Op()

    partials1, partials2 = diff._parse_partials(op)
    assert not partials1 and not partials2

    partials1, partials2 = diff._parse_partials(op, gradient=True)
    assert not partials2
    assert partials1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}

    partials1, partials2 = diff._parse_partials(op, gradient="x")
    assert not partials2
    assert partials1 == {(op, "x"): {"x": 1}}

    partials1, partials2 = diff._parse_partials(op, gradient={"z": {"x": 2, "y": 3}})
    assert not partials2
    assert partials1 == {"z": {"x": 2, "y": 3}}

    with pytest.raises(ValueError):
        diff._parse_partials(op, gradient="unknown")

    with pytest.raises(ValueError):
        diff._parse_partials(op, gradient={"z": {"unknown": 1}})

    partials1, partials2 = diff._parse_partials(op, hessian=True)
    assert partials1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}
    assert partials2 == {
        ((op, "x"), (op, "x")): {},
        ((op, "x"), (op, "y")): {},
        ((op, "y"), (op, "y")): {},
    }

    partials1, partials2 = diff._parse_partials(op, hessian="x")
    assert partials1 == {(op, "x"): {"x": 1}}
    assert partials2 == {((op, "x"), (op, "x")): {}}

    partials1, partials2 = diff._parse_partials(
        op,
        gradient={"foo": {"x": 1, "y": 2}, "bar": {"x": 1, "y": 2}},
        hessian={("foo", "bar"): {"x": 2, "y": 3}},
    )
    assert partials1 == {"foo": {"x": 1, "y": 2}, "bar": {"x": 1, "y": 2}}
    assert partials2 == {("foo", "bar"): {"x": 2, "y": 3}}

    with pytest.raises(ValueError):
        diff._parse_partials(op, hessian={("foo", "bar"): {"x": 2, "y": 3}})


def test_diffoperator_class():
    class MyOp(diff.DiffOperator):
        """dummy operator"""

        parameters = ["x", "y"]

        def _apply(self, sm):
            return sm

        def _derive1(self, variable, sm):
            coeff = {"x": 2, "y": 3}[variable]
            return coeff * sm

        def _derive2(self, pair, sm):
            pair = tuple(sorted(pair))
            coeff = {("x", "x"): 4, ("x", "y"): 5, ("y", "y"): 6}[pair]
            return coeff * sm

    op = MyOp(gradient=True)
    assert op.partials1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}
    assert not op.partials2
    assert op.diff1 == {"x", "y"}

    sm0 = core.StateMatrix([[[0, 0, 1]], [[1, 1, 0]], [[1 + 1j, 1 - 1j, 0]]])
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states)
    assert {(op, "x"), (op, "y")} == set(sm.gradient)
    assert np.allclose(sm.gradient[(op, "x")], 2 * sm0.states)
    assert np.allclose(sm.gradient[(op, "y")], 3 * sm0.states)

    op = MyOp(hessian=True)
    assert op.partials1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}
    assert op.partials2 == {
        ((op, "x"), (op, "x")): {},
        ((op, "x"), (op, "y")): {},
        ((op, "y"), (op, "y")): {},
    }
    assert op.diff1 == {"x", "y"}
    assert op.diff2 == {"x", "y"}

    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states)
    assert {(op, "x"), (op, "y")} == set(sm.gradient)
    assert np.allclose(sm.gradient[(op, "x")], 2 * sm0.states)
    assert np.allclose(sm.gradient[(op, "y")], 3 * sm0.states)
    assert np.allclose(sm.hessian[(op, "x"), (op, "x")], 4 * sm0.states)
    assert np.allclose(sm.hessian[(op, "x"), (op, "y")], 5 * sm0.states)
    assert np.allclose(sm.hessian[(op, "y"), (op, "y")], 6 * sm0.states)

    # custom combination of partial derivatives
    op = MyOp(gradient={"a": {"x": 0.1, "y": 0.2}}, hessian={("a", "a"): {"x": 0.3}})
    sm = op(sm0)
    assert np.allclose(sm.gradient["a"], (0.1 * 2 + 0.2 * 3) * sm0.states)
    assert np.allclose(
        sm.hessian[("a", "a")],
        (4 * 0.1 ** 2 + 2 * 5 * 0.1 * 0.2 + 6 * 0.2 ** 2 + 0.3 * 2) * sm0.states,
    )


def test_diff_E_class():
    # 0 state
    sm0 = core.StateMatrix([[1, 0, 1j], [1 + 1j, 1 - 1j, 0], [0, 1, -1j]])

    tau = 10
    T1 = 1e3
    T2 = 1e2
    g = 1e-1

    E = diff.E(tau, T1, T2, g=g, gradient=True)
    sm = E._apply(sm0.copy())

    # gradient w/r tau
    E_tau = diff.E(tau + 1e-8, T1, T2, g=g, gradient=True)
    fdiff = (E_tau._apply(sm0.copy()).states - sm.states) * 1e8
    sm_tau = E._derive1("tau", sm0.copy())
    assert np.allclose(sm_tau.states, fdiff)

    # gradient w/r T1
    E_T1 = diff.E(tau, T1 + 1e-8, T2, g=g, gradient=True)
    fdiff = (E_T1._apply(sm0.copy()).states - sm.states) * 1e8
    sm_T1 = E._derive1("T1", sm0.copy())
    assert np.allclose(sm_T1.states, fdiff)

    # gradient w/r T2
    E_T2 = diff.E(tau, T1, T2 + 1e-8, g=g, gradient=True)
    fdiff = (E_T2._apply(sm0.copy()).states - sm.states) * 1e8
    sm_T2 = E._derive1("T2", sm0.copy())
    assert np.allclose(sm_T2.states, fdiff)

    # gradient w/r g
    E_g = diff.E(tau, T1, T2, g=g + 1e-8, gradient=True)
    fdiff = (E_g._apply(sm0.copy()).states - sm.states) * 1e8
    sm_g = E._derive1("g", sm0.copy())
    assert np.allclose(sm_g.states, fdiff)

    #
    # 2nd order
    E = diff.E(tau, T1, T2, g=g, hessian=True)

    fdiff = (E_tau._derive1("tau", sm0.copy()).states - sm_tau.states) * 1e8
    assert np.allclose(E._derive2(("tau", "tau"), sm0.copy()).states, fdiff)

    fdiff = (E_T1._derive1("T1", sm0.copy()).states - sm_T1.states) * 1e8
    assert np.allclose(E._derive2(("T1", "T1"), sm0.copy()).states, fdiff)

    fdiff = (E_T2._derive1("T2", sm0.copy()).states - sm_T2.states) * 1e8
    assert np.allclose(E._derive2(("T2", "T2"), sm0.copy()).states, fdiff)

    fdiff = (E_g._derive1("g", sm0.copy()).states - sm_g.states) * 1e8
    assert np.allclose(E._derive2(("g", "g"), sm0.copy()).states, fdiff)

    fdiff = (E_tau._derive1("T1", sm0.copy()).states - sm_T1.states) * 1e8
    assert np.allclose(E._derive2(("tau", "T1"), sm0.copy()).states, fdiff)
    fdiff = (E_T1._derive1("tau", sm0.copy()).states - sm_tau.states) * 1e8
    assert np.allclose(E._derive2(("T1", "tau"), sm0.copy()).states, fdiff)

    fdiff = (E_tau._derive1("T2", sm0.copy()).states - sm_T2.states) * 1e8
    assert np.allclose(E._derive2(("tau", "T2"), sm0.copy()).states, fdiff)
    fdiff = (E_T2._derive1("tau", sm0.copy()).states - sm_tau.states) * 1e8
    assert np.allclose(E._derive2(("T2", "tau"), sm0.copy()).states, fdiff)

    fdiff = (E_tau._derive1("g", sm0.copy()).states - sm_g.states) * 1e8
    assert np.allclose(E._derive2(("tau", "g"), sm0.copy()).states, fdiff)
    fdiff = (E_g._derive1("tau", sm0.copy()).states - sm_tau.states) * 1e8
    assert np.allclose(E._derive2(("g", "tau"), sm0.copy()).states, fdiff)

    fdiff = (E_tau._derive1("T2", sm0.copy()).states - sm_T2.states) * 1e8
    assert np.allclose(E._derive2(("tau", "T2"), sm0.copy()).states, fdiff)
    fdiff = (E_T2._derive1("tau", sm0.copy()).states - sm_tau.states) * 1e8
    assert np.allclose(E._derive2(("T2", "tau"), sm0.copy()).states, fdiff)

    assert np.allclose(E._derive2(("T1", "T2"), sm0.copy()).states, 0)
    assert np.allclose(E._derive2(("T1", "g"), sm0.copy()).states, 0)
    assert np.allclose(E._derive2(("T2", "g"), sm0.copy()).states, 0)


def test_diff_T_class():
    T = diff.T(90, 90, gradient=True)

    # init state
    # sm0 = core.StateMatrix([[1 + 1j, 0, 0], [0, 0, 1], [0, 1 - 1j, 0]])
    sm0 = core.StateMatrix([[[0, 0, 1]], [[1, 1, 0]], [[1 + 1j, 1 - 1j, 0]]])
    sm = T._apply(sm0.copy())

    # gradient w/r alpha
    T_alpha = diff.T(90 + 1e-8, 90, gradient=True)
    fdiff = (T_alpha._apply(sm0.copy()).states - sm.states) * 1e8
    dsm_alpha = T._derive1("alpha", sm0.copy())
    assert np.allclose(dsm_alpha.states, fdiff)

    # gradient w/r phi
    T_phi = diff.T(90, 90 + 1e-8, gradient=True)
    fdiff = (T_phi._apply(sm0.copy()).states - sm.states) * 1e8
    dsm_phi = T._derive1("phi", sm0.copy())
    assert np.allclose(dsm_phi.states, fdiff)

    # 2nd derivatives
    T = diff.T(90, 90, hessian=True)
    fdiff2 = (T_alpha._derive1("alpha", sm0.copy()).states - dsm_alpha.states) * 1e8
    assert np.allclose(fdiff2, T._derive2(("alpha", "alpha"), sm0.copy()).states)

    fdiff2 = (T_phi._derive1("phi", sm0.copy()).states - dsm_phi.states) * 1e8
    assert np.allclose(fdiff2, T._derive2(("phi", "phi"), sm0.copy()).states)

    fdiff2 = (T_phi._derive1("alpha", sm0.copy()).states - dsm_alpha.states) * 1e8
    assert np.allclose(fdiff2, T._derive2(("alpha", "phi"), sm0.copy()).states)

    fdiff2 = (T_alpha._derive1("phi", sm0.copy()).states - dsm_phi.states) * 1e8
    assert np.allclose(fdiff2, T._derive2(("phi", "alpha"), sm0.copy()).states)


def test_diff_chain():
    excit = diff.T(90, 90, name="excit")
    refoc = diff.T(150, 0, gradient="alpha", name="refoc")
    relax = diff.E(5, 1e3, 35, gradient="T2", name="relax")
    grad = diff.S(1, name="grad")
    necho = 5
    spinecho = [excit] + [grad, relax, refoc, grad, relax] * necho

    # check partial derivative definitions
    assert refoc.diff1 == {"alpha"}
    assert relax.diff1 == {"T2"}

    # finite differences
    _relax = diff.E(5, 1e3, 35 + 1e-8)
    spinecho_T2 = [excit] + [grad, _relax, refoc, grad, _relax] * necho

    _refoc = diff.T(150 + 1e-8, 0)
    spinecho_alpha = [excit] + [grad, relax, _refoc, grad, relax] * necho

    # simulate
    sm = core.StateMatrix([0, 0, 1])
    sm_T2 = sm.copy()
    sm_alpha = sm.copy()
    states = []
    for i, op in enumerate(spinecho):
        # update state matrix and gradients
        sm = op(sm)
        sm_T2 = spinecho_T2[i](sm_T2)
        sm_alpha = spinecho_alpha[i](sm_alpha)
        states.append(sm)

    # compare to finite difference
    key_T2 = (relax, "T2")
    assert np.allclose(
        (sm_T2.states - sm.states) * 1e8, sm.gradient[key_T2].states, atol=1e-7
    )
    key_alpha = (refoc, "alpha")
    assert np.allclose(
        (sm_alpha.states - sm.states) * 1e8, sm.gradient[key_alpha].states, atol=1e-7
    )

    # using simulate
    probe = ["F0", diff.Jacobian(key_T2), diff.Jacobian(key_alpha)]
    spinecho = [excit] + [grad, relax, refoc, grad, relax, core.ADC] * necho
    signal, gradT2, gradalpha = core.simulate(
        spinecho,
        init=core.StateMatrix([0, 0, 1]),
        probe=probe,
    )
    assert np.allclose(signal[-1], sm.F0)
    assert np.allclose(gradT2[-1], sm.gradient[key_T2].F0)
    assert np.allclose(gradalpha[-1], sm.gradient[key_alpha].F0)


def test_diff2_chain():
    """test 2nd order differentials"""
    alpha, phi = 20, 90
    tau, T1, T2 = 5, 1e3, 30

    pulse = diff.T(alpha, phi, name="pulse", gradient=True, hessian="alpha")
    relax = diff.E(tau, T1, T2, name="relax", hessian="T2")
    grad = diff.S(1, name="grad")
    necho = 5
    seq = [pulse, relax, grad] * necho

    # finite difference operators
    _pulse = diff.T(alpha + 1e-8, phi, gradient="alpha")
    _relax = diff.E(tau, T1, T2 + 1e-8, gradient="T2")

    # simulate
    sm = core.StateMatrix([0, 0, 1])
    sm_T2 = sm.copy()
    sm_alpha = sm.copy()
    for i, op in enumerate(seq):
        sm = op(sm)
        sm_T2 = _relax(sm_T2) if op.name == "relax" else op(sm_T2)
        sm_alpha = _pulse(sm_alpha) if op.name == "pulse" else op(sm_alpha)

    # alpha: compare to finite difference
    fdiff1 = (sm_alpha.states - sm.states) * 1e8
    assert np.allclose(fdiff1, sm.gradient[(pulse, "alpha")])

    fdiff2 = (
        sm_alpha.gradient[(_pulse, "alpha")].states
        - sm.gradient[(pulse, "alpha")].states
    ) * 1e8
    assert np.allclose(fdiff2, sm.hessian[((pulse, "alpha"), (pulse, "alpha"))].states)
    F0_alpha_alpha = sm.hessian[((pulse, "alpha"), (pulse, "alpha"))].F0

    # T2: compare to finite difference
    fdiff1 = (sm_T2.states - sm.states) * 1e8
    assert np.allclose(fdiff1, sm.gradient[(relax, "T2")])

    fdiff2 = (
        sm_T2.gradient[(_relax, "T2")].states - sm.gradient[(relax, "T2")].states
    ) * 1e8
    assert np.allclose(fdiff2, sm.hessian[((relax, "T2"), (relax, "T2"))].states)
    F0_T2_T2 = sm.hessian[((relax, "T2"), (relax, "T2"))].F0

    # T2/alpha
    fdiff2 = (
        sm_T2.gradient[(pulse, "alpha")].states - sm.gradient[(pulse, "alpha")].states
    ) * 1e8
    assert np.allclose(fdiff2, sm.hessian[((relax, "T2"), (pulse, "alpha"))].states)
    assert np.allclose(fdiff2, sm.hessian[((pulse, "alpha"), (relax, "T2"))].states)
    # alpha/T2
    fdiff2 = (
        sm_alpha.gradient[(relax, "T2")].states - sm.gradient[(relax, "T2")].states
    ) * 1e8
    assert np.allclose(fdiff2, sm.hessian[((relax, "T2"), (pulse, "alpha"))].states)
    F0_alpha_T2 = sm.hessian[((pulse, "alpha"), (relax, "T2"))].F0

    # using simulate
    probe = [
        lambda sm: sm.hessian[((pulse, "alpha"), (pulse, "alpha"))].F0,
        lambda sm: sm.hessian[((pulse, "alpha"), (relax, "T2"))].F0,
        lambda sm: sm.hessian[((relax, "T2"), (relax, "T2"))].F0,
    ]
    hessian = core.simulate(
        seq + [core.ADC],
        probe=probe,
        squeeze=False,  # tmp
    )
    assert np.isclose(hessian[0], F0_alpha_alpha)
    assert np.isclose(hessian[1], F0_alpha_T2)
    assert np.isclose(hessian[2], F0_T2_T2)


def test_A_combine():
    """test combining B operators"""
    E = diff.E(5, 1e2, 5e2, g=1e-1, gradient="T2", name="E")
    Ta = diff.T(15, 90, gradient={"b1": {"alpha": 15}, "phi": {"phi": 1}}, name="Ta")
    Tb = diff.T(20, 90, gradient={"b1": {"alpha": 20}, "phi": {"phi": 1}}, name="Tb")

    assert E.diff1 == {"T2"}
    assert Ta.diff1 == {"alpha", "phi"}
    assert Tb.diff1 == {"alpha", "phi"}

    # combine operators
    ops = [Ta, E, Tb, E, Ta, E, Tb, E]
    combined = diff.A.combine(ops)
    assert combined.diff1 == {
        (E, "T2"),
        "b1",
        "phi",
    }

    # simulate separately
    sm0 = core.StateMatrix([1, 1, 0])
    sm1 = sm0
    for i, op in enumerate(ops):
        sm1 = op(sm1)

    sm2 = core.StateMatrix([1, 1, 0])
    sm2 = combined(sm2)
    # assert set(sm2.gradient) == combined.diff1
    assert set(sm2.gradient) == set(combined.partials1)

    assert np.allclose(sm1.states, sm2.states)
    for partial in combined.partials1:
        assert np.allclose(sm1.gradient[partial].states, sm2.gradient[partial])

    # 2nd order combine
    E = diff.E(5, 1e2, 5e2, g=1e-1, hessian="T2", name="E")
    Ta = diff.T(15, 90, hessian=True, name="Ta")
    Tb = diff.T(20, 90, hessian=True, name="Tb")

    assert E.diff2 == {"T2"}
    assert E.auto_cross_derivatives
    assert Ta.diff2 == {"alpha", "phi"}
    assert Ta.auto_cross_derivatives
    assert Tb.diff2 == {"alpha", "phi"}
    assert Tb.auto_cross_derivatives

    # combine operators
    ops = [Ta, E, Tb, E, Ta, E, Tb, E]
    combined = diff.A.combine(ops)
    assert combined.diff2 == {
        (E, "T2"),
        (Ta, "alpha"),
        (Ta, "phi"),
        (Tb, "alpha"),
        (Tb, "phi"),
    }

    # simulate separately
    sm0 = core.StateMatrix([1, 1, 0])
    sm1 = sm0
    for i, op in enumerate(ops):
        sm1 = op(sm1)

    sm2 = core.StateMatrix([1, 1, 0])
    sm2 = combined(sm2)

    assert set(sm2.gradient) == set(sm1.gradient)
    for var in sm1.gradient:
        assert np.allclose(sm1.gradient[var].states, sm2.gradient[var].states)

    assert set(sm2.hessian) == set(sm1.hessian)
    for pair in sm1.hessian:
        assert np.allclose(sm1.hessian[pair].states, sm2.hessian[pair].states)


def test_combine_2():
    E = diff.E(10, 1e3, 35, hessian=["T1", "T2"], name="E")
    T = diff.T(150, 0, hessian=["alpha"], name="T")

    sm0 = core.StateMatrix([1, 1, 0])
    sm1 = E(T(E(sm0)))

    combined = diff.A.combine([E, T, E])
    sm2 = combined(sm0)

    assert np.allclose(sm1.states, sm2.states)
    assert set(sm1.gradient) == set(sm2.gradient)
    for var in sm1.gradient:
        assert np.allclose(sm1.gradient[var], sm2.gradient[var])
    assert set(sm1.hessian) == set(sm2.hessian)
    pairs = list(sm1.hessian)
    for var in sm1.hessian:
        assert np.allclose(sm1.hessian[var], sm2.hessian[var])

    # repeat combined
    sm1 = E(T(E(sm0)))
    smx = combined(sm0)

    _combined = diff.A.combine([T, E])
    combined2 = diff.A.combine([E, _combined])
    sm2 = combined2(sm0)

    assert np.allclose(sm1.states, sm2.states)
    assert set(sm1.gradient) == set(sm2.gradient)
    for var in sm1.gradient:
        assert np.allclose(sm1.gradient[var], sm2.gradient[var])
    assert set(sm1.hessian) == set(sm2.hessian)
    pairs = list(sm1.hessian)
    for var in sm1.hessian:
        assert np.allclose(sm1.hessian[var], sm2.hessian[var])


def test_jacobian_class():
    # jacobian probe
    necho = 5
    rf = diff.T(15, 90, gradient=["alpha"])
    relax = diff.E(5, 1e3, 30, gradient=["T2"])
    shift = diff.S(1)
    adc = core.ADC

    seq = [rf, relax, shift, adc] * necho

    probes = [
        diff.Jacobian([(rf, "alpha")]),
        diff.Jacobian([(rf, "alpha"), (relax, "T2")]),
        diff.Jacobian(["magnitude", (rf, "alpha")]),
    ]

    sm = core.StateMatrix()
    sms = []
    for op in seq:
        sm = op(sm)
        if isinstance(op, core.Probe):
            sms.append(sm)

    jac1, jac2, jac3 = core.simulate(seq, probe=probes)
    assert jac1.shape == (5, 1, 1)  # alpha
    assert np.allclose(jac1[-1], sm.gradient[(rf, "alpha")].F0)

    assert jac2.shape == (5, 2, 1)  # alpha, T2
    assert np.allclose(jac2[-1, 0], sm.gradient[(rf, "alpha")].F0)
    assert np.allclose(jac2[-1, 1], sm.gradient[(relax, "T2")].F0)

    assert jac3.shape == (5, 2, 1)  # magnitude, alpha
    assert np.allclose(jac3[-1, 0], sm.F0)
    assert np.allclose(jac3[-1, 1], sm.gradient[(rf, "alpha")].F0)


def test_hessian_class():
    # hessian probe
    necho = 5
    rf = diff.T(15, 90, hessian=["alpha"], name="T")
    relax = diff.E(5, 1e3, 30, hessian=["T2"], name="E")
    shift = diff.S(1, name="S")
    adc = core.ADC

    seq = [rf, relax, shift, adc] * necho

    probes = [
        diff.Hessian((rf, "alpha")),
        diff.Hessian([(rf, "alpha"), (relax, "T2")]),
        diff.Hessian(["magnitude", (rf, "alpha")], (relax, "T2")),
    ]

    sm = core.StateMatrix()
    tmp = []
    for op in seq:
        sm = op(sm, inplace=False)
        if isinstance(op, core.Probe):
            tmp.append(probes[0].acquire(sm))

    hes1, hes2, hes3 = core.simulate(seq, probe=probes, squeeze=False)

    assert hes1.shape == (necho, 1, 1, 1)  # alpha
    assert np.allclose(hes1[-1, 0, 0], sm.hessian[((rf, "alpha"), (rf, "alpha"))].F0)

    assert hes2.shape == (necho, 2, 2, 1)  # alpha, T2
    assert np.allclose(hes2[-1, 0, 0], sm.hessian[((rf, "alpha"), (rf, "alpha"))].F0)
    assert np.allclose(hes2[-1, 0, 1], sm.hessian[((relax, "T2"), (rf, "alpha"))].F0)
    assert np.allclose(hes2[-1, 1, 0], sm.hessian[((rf, "alpha"), (relax, "T2"))].F0)
    assert np.allclose(hes2[-1, 1, 1], sm.hessian[((relax, "T2"), (relax, "T2"))].F0)

    assert hes3.shape == (necho, 2, 1, 1)  # (magnitude, alpha) x T2
    assert np.allclose(hes3[-1, 0, 0], sm.gradient[(relax, "T2")].F0)  # magnitude
    assert np.allclose(hes3[-1, 1, 0], sm.hessian[((rf, "alpha"), (relax, "T2"))].F0)


def test_partial_hessian():
    # partial hessian
    necho = 2
    rf = diff.T(
        15,
        90,
        gradient={"alpha": {"alpha": 1}},
        hessian={("alpha", "T2"): {}, ("alpha", "T1"): {}},
        name="rf1",
    )
    relax = diff.E(
        5,
        1e3,
        30,
        gradient={"T2": {"T2": 1}, "T1": {"T1": 1}},
        hessian={("alpha", "T2"): {}, ("alpha", "T1"): {}},
        name="relax1",
    )
    shift = diff.S(1, name="shift1")
    adc = core.ADC

    seq = [rf, relax, shift, adc] * necho

    sm = core.StateMatrix()
    for op in seq:
        sm = op(sm)

    assert set(sm.hessian) == {
        ("alpha", "T2"),
        ("alpha", "T1"),
        ("T1", "alpha"),
        ("T2", "alpha"),
    }

    # full hessian
    rf_ = diff.T(15, 90, hessian=True, name="rf2")
    relax_ = diff.E(5, 1e3, 30, hessian=True, name="relax2")
    shift = diff.S(1, name="shift2")
    seq_ = [rf_, relax_, shift, adc] * necho
    sm2 = core.StateMatrix()
    for op in seq_:
        sm2 = op(sm2)
    assert np.allclose(
        sm2.hessian[(rf_, "alpha"), (relax_, "T2")], sm.hessian[("alpha", "T2")]
    )

    # finite diff
    rf_ = diff.T(15 + 1e-8, 90)
    seq_ = [rf_, relax, shift, adc] * necho
    sm_ = core.StateMatrix()
    for op in seq_:
        sm_ = op(sm_)
    fdiff = (sm_.gradient["T2"].F0 - sm.gradient["T2"].F0) * 1e8
    assert np.allclose(fdiff, sm.hessian[("alpha", "T2")].F0)


def test_pruning():
    necho = 50

    shift = diff.S(1)
    relax = diff.E(5, 50, 5, gradient=["T2"])
    adc = core.ADC

    def rf(i):
        hessian = "alpha" if i == 0 else None
        return diff.T(5, i ** 2, hessian=hessian, name=f"T_{i:02}")

    seq = [[rf(i), relax, adc, relax, shift] for i in range(necho)]
    variables = [(relax, "T2"), (seq[0][0], "alpha")]

    # no pruning
    probe = [diff.Jacobian(variables), diff.Hessian(variables[1])]
    jac1, hes1 = core.simulate(seq, probe=probe)
    assert jac1.shape == (necho, len(variables), 1)

    # with pruning
    pruner = diff.PartialsPruner(threshold=1e-5, variables=[variables[1]])
    jac2, hes2 = core.simulate(seq, probe=probe, callback=pruner)

    assert not any(np.isclose(jac2[:, 0], 0, atol=1e-9))
    assert not any(np.isclose(jac1[:, 1], 0, atol=1e-9))
    assert any(np.isclose(jac2[:, 1], 0, atol=1e-9))
    assert np.allclose(jac1, jac2, atol=1e-6)

    assert not any(np.isclose(hes1, 0, atol=1e-12))
    assert any(np.isclose(hes2, 0, atol=1e-12))
    assert np.allclose(hes1, hes2, atol=1e-6)
