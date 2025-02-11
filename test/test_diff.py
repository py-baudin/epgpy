"""unittest for epgpy.diff"""

import pytest
import numpy as np
from epgpy import statematrix, diff, operators, functions


def test_parse_partials():

    class Op(diff.DiffOperator):
        PARAMETERS_ORDER1 = {"x", "y"}
        PARAMETERS_ORDER2 = {("x", "y"), ("x", "x")}  # no (y, y)

        def _apply(self, sm):
            pass

        def _derive1(self, *args):
            pass

        def _derive2(self, sm, params):
            pass

        def __repr__(self):
            return "op"

    # dummy operator
    op = Op()

    order1, order2 = op._parse_partials()
    assert not order1 and not order2

    # all order1 derivatives
    order1, order2 = op._parse_partials(order1=True)
    assert not order2
    assert order1 == {"x": {"x": 1}, "y": {"y": 1}}

    # selected order1 derivatives
    order1, order2 = op._parse_partials(order1="x")
    assert not order2
    assert order1 == {"x": {"x": 1}}

    # selected aliased order1 derivatives
    order1, order2 = op._parse_partials(order1={"x1": "x"})
    assert not order2
    assert order1 == {"x1": {"x": 1}}

    with pytest.raises(ValueError):
        # unknown parameter
        op._parse_partials(order1="unknown")

    with pytest.raises(ValueError):
        # unknown parameter
        op._parse_partials(order1={"x1": "z"})

    # order 2
    order1, order2 = op._parse_partials(order2=True)
    assert order1 == {"x": {"x": 1}, "y": {"y": 1}}
    assert order2 == {
        ("x", "x"): {},
        ("x", "y"): {},
    }

    order1, order2 = op._parse_partials(order1="x", order2="x")
    assert order1 == {"x": {"x": 1}}
    assert order2 == {("x", "x"): {}}

    order1, order2 = op._parse_partials(order1=["x", "y"], order2=[("x", "y")])
    assert order1 == {"x": {"x": 1}, "y": {"y": 1}}
    assert order2 == {("x", "y"): {}}

    order1, order2 = op._parse_partials(
        order1={"x1": "x", "y1": "y"},
        order2=[("x1", "y1")],
    )
    assert order1 == {"x1": {"x": 1}, "y1": {"y": 1}}
    assert order2 == {("x1", "y1"): {}}

    order1, order2 = op._parse_partials(
        order1={"x1": {"x": 2}, "y1": {"y": 3}},
        order2={("y1", "x1"): {"x": 4, "y": 5}},
    )
    assert order1 == {"x1": {"x": 2}, "y1": {"y": 3}}
    assert order2 == {("x1", "y1"): {"x": 4, "y": 5}}

    with pytest.raises(ValueError):
        # order1 not given
        op._parse_partials(order2={("x", "y")})

    with pytest.raises(ValueError):
        # unknown variable pair
        op._parse_partials(order1=["x", "y"], order2=[("a", "b")])

    # with pytest.raises(ValueError):
    with pytest.warns(UserWarning):
        # invalid pair
        op._parse_partials(order1=["x", "y"], order2=[("y", "y")])

    with pytest.warns(UserWarning):
        # invalid pair
        op._parse_partials(order1={"a": "x", "b": "y"}, order2=[("b", "b")])

    with pytest.raises(ValueError):
        # unknown parameter
        op._parse_partials(order1={"a": "x", "b": "y"}, order2={("a", "b"): {"z": 1}})

    with pytest.raises(ValueError):
        # invalid pair: expecting no coefficients in cross-operator pairs
        op._parse_partials(
            order1={"a": {"x": 2}, "b": {"y": 3}},
            order2={("a", "c"): {"x": 4}},
        )


def test_order12():
    """Test order 1 and 2 partials"""

    class Op(diff.DiffOperator):
        PARAMETERS_ORDER1 = {"x", "y"}
        PARAMETERS_ORDER2 = {("x", "y"), ("x", "x")}  # no (y, y)

        def _apply(self, sm):
            return sm

        def _derive1(self, sm, param):
            op1 = {"x": 2, "y": 3}[param]
            return op1 * sm

        def _derive2(self, sm, pair):
            pair = tuple(sorted(pair))
            op2 = {("x", "x"): 4, ("x", "y"): 5}[pair]
            return op2 * sm

    # order 1
    op = Op(order1=True)
    assert op.order1 == {"x": {"x": 1}, "y": {"y": 1}}
    assert not op.order2
    assert op.parameters_order1 == {"x", "y"}

    # nd state matrix
    sm0 = statematrix.StateMatrix([[[0, 0, 1]], [[1, 1, 0]], [[1 + 1j, 1 - 1j, 0]]])
    # apply operator
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states)  # sm is unchanged
    assert set(sm.order1) == {"x", "y"}  # order1 partials were computed
    assert np.allclose(sm.order1["x"], 2 * sm0.states)
    assert np.allclose(sm.order1["y"], 3 * sm0.states)

    # composed arguments
    op = Op(order1={"z": {"x": -1, "y": -2}})
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states)  # sm is unchanged
    assert set(sm.order1) == {"z"}
    assert np.allclose(sm.order1["z"], (-1 * 2 + -2 * 3) * sm0.states)

    # order 2

    op = Op(order2=True)
    # order 1 is filled by default
    assert op.order1 == {"x": {"x": 1}, "y": {"y": 1}}
    assert op.order2 == {("x", "x"): {}, ("x", "y"): {}}
    assert op.parameters_order1 == {"x", "y"}
    assert op.parameters_order2 == {("x", "y"), ("x", "x")}

    # apply operator
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states)  # sm is unchanged
    # order1 partials were computed
    assert set(sm.order1) == {"x", "y"}
    # order2 partials were computed
    assert set(sm.order2) == {("x", "x"), ("x", "y"), ("y", "x")}
    assert np.allclose(sm.order1["x"], 2 * sm0.states)
    assert np.allclose(sm.order1["y"], 3 * sm0.states)
    assert np.allclose(sm.order2["x", "x"], 4 * sm0.states)
    assert np.allclose(sm.order2["x", "y"], 5 * sm0.states)

    # composed arguments
    op = Op(
        order1={"z": {"x": -1}, "y": {"y": 1}},
        order2={("z", "z"): {"x": -2}, ("y", "z"): {}},
    )
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states)  # sm is unchanged
    assert set(sm.order1) == {"z", "y"}
    assert set(sm.order2) == {("y", "z"), ("z", "y"), ("z", "z")}
    assert np.allclose(sm.order1["z"], (-1 * 2) * sm0.states)
    assert np.allclose(sm.order1["y"], 3 * sm0.states)
    assert np.allclose(sm.order2[("y", "z")], (1 * -1 * 5) * sm0.states)
    assert np.allclose(sm.order2[("z", "z")], (-1 * -1 * 4 + -2 * 2) * sm0.states)

    # test copy
    op2 = op.copy(name="op2")
    assert isinstance(op2, type(op))
    assert op2.name == "op2"
    assert op2.order1 == op.order1
    assert op2.PARAMETERS_ORDER1 == op.PARAMETERS_ORDER1
    assert op2.order2 == op.order2
    assert op2.PARAMETERS_ORDER2 == op.PARAMETERS_ORDER2


def test_diff_chain():
    """chain multiple differentiable operators"""

    class Op(diff.DiffOperator):
        PARAMETERS_ORDER1 = {"x", "y"}
        PARAMETERS_ORDER2 = {("x", "y"), ("x", "x"), ("y", "y")}

        def _apply(self, sm):
            return sm

        def _derive1(self, sm, param):
            op1 = {"x": 2, "y": 3}[param]
            return op1 * sm

        def _derive2(self, sm, pair):
            pair = tuple(sorted(pair))
            op2 = {("x", "x"): 4, ("x", "y"): 5, ("y", "y"): 6}[pair]
            return op2 * sm

    # order 1, aliases
    op1 = Op(order1={"a": "x", "b": "y"})
    op2 = Op(order1={"a": "y", "b": "x"})
    sm0 = statematrix.StateMatrix([1, 1, 0.5])
    sm2 = op2(op1(sm0))
    assert set(sm2.order1) == {"a", "b"}
    assert np.allclose(sm2.states, sm0.states)  # order 0
    # op2/d1 * op1 + op2 * op1/d1
    assert np.allclose(sm2.order1["a"].states, (2 + 3) * sm0.states)
    assert np.allclose(sm2.order1["b"].states, (3 + 2) * sm0.states)

    # order1, composed
    op1 = Op(order1={"a": {"x": 1, "y": 2}, "b": {"y": 3}})
    op2 = Op(order1={"a": {"x": 1}, "b": {"x": 2, "y": 3}})
    sm0 = statematrix.StateMatrix([1, 1, 0.5])
    sm2 = op2(op1(sm0))
    assert set(sm2.order1) == {"a", "b"}
    assert np.allclose(sm2.states, sm0.states)  # order 0
    assert np.allclose(sm2.order1["a"].states, ((1 * 2 + 2 * 3) + 2) * sm0.states)
    assert np.allclose(sm2.order1["b"].states, ((3 * 3) + (2 * 2 + 3 * 3)) * sm0.states)

    # order 2
    op1 = Op(order1={"a": "x", "b": "y"}, order2=[("a", "b")])
    sm0 = statematrix.StateMatrix([1, 1, 0.5])
    sm1 = op1(sm0)
    assert set(sm1.order2) == {("a", "b"), ("b", "a")}
    assert np.allclose(sm1.order2[("a", "b")].states, 5 * sm0.states)
    assert np.allclose(sm1.order2[("a", "b")].states, sm1.order2[("b", "a")].states)

    op2 = Op(order1={"a": "y", "b": "x"}, order2=[("a", "b")])
    sm2 = op2(sm1)
    assert set(sm2.order2) == {("a", "b"), ("b", "a")}
    assert np.allclose(
        sm2.order2[("a", "b")].states,
        (5 + 3 * 3 + 2 * 2 + 5) * sm0.states,
    )
    assert np.allclose(sm2.order2[("a", "b")].states, sm2.order2[("b", "a")])

    # order2, composed
    op1 = Op(order1={"a": {"x": 1, "y": 2}, "b": {"y": 3}}, order2=[("a", "b")])
    op2 = Op(order1={"a": {"x": 1}, "b": {"x": 2, "y": 3}}, order2=[("a", "b")])
    sm0 = statematrix.StateMatrix([1, 1, 0.5])
    sm1 = op1(sm0)
    assert set(sm1.order2) == {("a", "b"), ("b", "a")}
    assert np.allclose(sm1.order2[("a", "b")], (1 * 3 * 5 + 2 * 3 * 6) * sm0.states)

    sm2 = op2(sm1)
    assert set(sm2.order2) == {("a", "b"), ("b", "a")}
    assert np.allclose(
        sm2.order2[("a", "b")],
        # 2nd derivatives
        (
            (1 * 3 * 5 + 2 * 3 * 6)
            + (1 * 2 * 4 + 1 * 3 * 5)
            +
            # cross derivatives
            (1 * 2) * (3 * 3)
            + (2 * 2 + 3 * 3) * (1 * 2 + 2 * 3)
        )
        * sm0.states,
    )


def test_diff_chain_mse():
    exc = operators.T(90, 90, name="exc")
    ref = operators.T(150, 0, order1="alpha", name="ref")
    relax = operators.E(5, 1e3, 35, order1="T2", name="relax")
    grad = operators.S(1, name="grad")
    necho = 5
    seq = [exc] + [grad, relax, ref, grad, relax] * necho

    # check partial derivative definitions
    assert ref.parameters_order1 == {"alpha"}
    assert relax.parameters_order1 == {"T2"}

    # finite differences
    rlx_T2 = operators.E(5, 1e3, 35 + 1e-8)
    ref_alpha = operators.T(150 + 1e-8, 0)

    # simulate
    sm = statematrix.StateMatrix([0, 0, 1])
    sm_T2 = sm.copy()
    sm_alpha = sm.copy()
    states = []
    for i, op in enumerate(seq):
        # update state matrix and gradients
        sm = op(sm)
        sm_T2 = rlx_T2(sm_T2) if op.name == "relax" else op(sm_T2)
        sm_alpha = ref_alpha(sm_alpha) if op.name == "ref" else op(sm_alpha)
        states.append(sm)

    # check derivatives
    assert set(sm.order1) == {"T2", "alpha"}

    # compare to finite difference
    assert np.allclose(
        (sm_T2.states - sm.states) * 1e8, sm.order1["T2"].states, atol=1e-7
    )
    assert np.allclose(
        (sm_alpha.states - sm.states) * 1e8, sm.order1["alpha"].states, atol=1e-7
    )

    # using simulate
    probe = ["F0", diff.Jacobian("T2"), diff.Jacobian("alpha")]
    spinecho = [exc] + [grad, relax, ref, grad, relax, operators.ADC] * necho
    signal, gradT2, gradalpha = functions.simulate(
        spinecho,
        init=[0, 0, 1],
        probe=probe,
    )
    assert np.allclose(signal[-1], sm.F0)
    assert np.allclose(gradT2[-1], sm.order1["T2"].F0)
    assert np.allclose(gradalpha[-1], sm.order1["alpha"].F0)


def test_diff2_ssfp():
    """test 2nd order differentials"""
    alpha, phi = 20, 90
    tau, T1, T2 = 5, 1e3, 30

    rf = operators.T(alpha, phi, name="pulse", order2="alpha")
    rlx = operators.E(tau, T1, T2, name="relax", order2="T2")
    grd = operators.S(1, name="grad")
    necho = 5
    seq = [rf, rlx, grd] * necho

    # finite difference operators
    rf_alpha = operators.T(alpha + 1e-8, phi, order1="alpha")
    rlx_T2 = operators.E(tau, T1, T2 + 1e-8, order1="T2")

    # simulate
    sm = statematrix.StateMatrix([0, 0, 1])
    sm_T2 = sm.copy()
    sm_alpha = sm.copy()
    for op in seq:
        sm = op(sm)
        sm_T2 = rlx_T2(sm_T2) if op.name == "relax" else op(sm_T2)
        sm_alpha = rf_alpha(sm_alpha) if op.name == "pulse" else op(sm_alpha)

    assert set(sm.order1) == {"alpha", "T2"}
    assert set(sm.order2) == {
        ("alpha", "alpha"),
        ("T2", "T2"),
        ("alpha", "T2"),
        ("T2", "alpha"),
    }

    # finite difference

    # order1
    assert np.allclose(
        (sm_alpha.states - sm.states) * 1e8, sm.order1["alpha"].states, atol=1e-7
    )
    assert np.allclose(
        (sm_T2.states - sm.states) * 1e8, sm.order1["T2"].states, atol=1e-7
    )

    # order2
    assert np.allclose(
        (sm_alpha.order1["alpha"].states - sm.order1["alpha"].states) * 1e8,
        sm.order2[("alpha", "alpha")].states,
    )

    assert np.allclose(
        (sm_alpha.order1["T2"].states - sm.order1["T2"].states) * 1e8,
        sm.order2[("T2", "alpha")].states,
    )

    assert np.allclose(
        (sm_T2.order1["alpha"].states - sm.order1["alpha"].states) * 1e8,
        sm.order2[("T2", "alpha")].states,
    )

    assert np.allclose(
        (sm_T2.order1["T2"].states - sm.order1["T2"].states) * 1e8,
        sm.order2[("T2", "T2")].states,
    )

    # using simulate
    probe = [
        lambda sm: sm.order2[("alpha", "alpha")].F0,
        lambda sm: sm.order2[("T2", "alpha")].F0,
        lambda sm: sm.order2[("T2", "T2")].F0,
    ]
    order2 = functions.simulate(
        seq + [operators.ADC],
        probe=probe,
    )
    assert np.isclose(order2[0], sm.order2[("alpha", "alpha")].F0)
    assert np.isclose(order2[1], sm.order2[("T2", "alpha")].F0)
    assert np.isclose(order2[2], sm.order2[("T2", "T2")].F0)


def test_diff2_partial():
    # partial order2
    necho = 2
    rf1 = operators.T(
        15,
        90,
        order1="alpha",
        order2=[("alpha", "T2"), ("alpha", "T1")],
        name="rf1",
    )
    rlx1 = operators.E(
        5,
        1e3,
        30,
        order1=["T2", "T1"],
        order2=[("alpha", "T2"), ("alpha", "T1")],
        name="rlx1",
    )
    grd1 = operators.S(1, name="grd1")
    adc = operators.ADC
    seq1 = [rf1, rlx1, grd1, adc] * necho

    sm1 = statematrix.StateMatrix()
    for op in seq1:
        sm1 = op(sm1)

    assert set(sm1.order2) == {
        ("alpha", "T2"),
        ("alpha", "T1"),
        ("T1", "alpha"),
        ("T2", "alpha"),
    }

    # full order2
    rf2 = operators.T(15, 90, order2=True, name="rf2")
    rlx2 = operators.E(5, 1e3, 30, order2=True, name="rlx2")
    grd2 = operators.S(1, name="grd2")
    seq2 = [rf2, rlx2, grd2, adc] * necho

    sm2 = statematrix.StateMatrix()
    for op in seq2:
        sm2 = op(sm2)
    assert np.allclose(sm2.order2[("T2", "alpha")], sm1.order2[("alpha", "T2")])

    # finite diff
    rf_alpha = operators.T(15 + 1e-8, 90)
    sm_alpha = statematrix.StateMatrix()
    for op in [rf_alpha, rlx1, grd1, adc] * necho:
        sm_alpha = op(sm_alpha)
    assert np.allclose(
        (sm_alpha.order1["T2"].F0 - sm1.order1["T2"].F0) * 1e8,
        sm1.order2[("T2", "alpha")].F0,
    )
    assert np.allclose(
        (sm_alpha.order1["T1"].F0 - sm1.order1["T1"].F0) * 1e8,
        sm1.order2[("T1", "alpha")].F0,
    )


def test_diff_combine():
    """test differential combined operators"""

    # sinc pulse
    npoint = 100
    nlobe = 5
    pulse = np.sinc(nlobe * np.linspace(-1, 1, npoint))

    # RF power
    alpha = 90
    pow = alpha / 180 / np.abs(np.sum(pulse))
    angles = pulse * pow / 180

    T, E = operators.T, operators.E
    rlx = E(1, 100, 10, order1=["T2", "g"], name="rlx")
    seq = [op for a in angles for op in [T(a, 0), rlx]]

    # finite diffs
    rlx_T2 = E(1, 100, 10 + 1e-8)
    rlx_g = E(1, 100, 10, g=1e-8)

    # loop
    sm0 = statematrix.StateMatrix()
    sm, sm_T2, sm_g = None, None, None
    for op in seq:
        sm = op(sm or sm0)
        sm_T2 = rlx_T2(sm_T2) if op.name == "rlx" else op(sm_T2 or sm0)
        sm_g = rlx_g(sm_g) if op.name == "rlx" else op(sm_g or sm0)

    assert set(sm.order1) == {"T2", "g"}
    assert np.allclose((sm_T2.states - sm.states) * 1e8, sm.order1["T2"].states)
    assert np.allclose((sm_g.states - sm.states) * 1e8, sm.order1["g"].states)

    # combine
    combined = seq[0]
    for op in seq[1:]:
        combined = combined @ op
    smc = combined(sm0)

    assert np.allclose(sm.states, smc.states)
    assert np.allclose(sm.order1["T2"].states, smc.order1["T2"].states)
    assert np.allclose(sm.order1["g"].states, smc.order1["g"].states)


def test_jacobian_class():
    # jacobian probe
    necho = 5
    rf = operators.T(15, 90, order1=["alpha"])
    rlx = operators.E(5, 1e3, 30, order1=["T2"])
    grd = operators.S(1)
    adc = operators.ADC

    seq = [rf, rlx, grd, adc] * necho

    probes = [
        diff.Jacobian(["alpha"]),
        diff.Jacobian(["alpha", "T2"]),
        diff.Jacobian(["magnitude", "alpha"]),
    ]

    sm = statematrix.StateMatrix()
    sms = []
    for op in seq:
        sm = op(sm)
        if isinstance(op, operators.Probe):
            sms.append(sm)

    jac1, jac2, jac3 = functions.simulate(seq, probe=probes)
    assert jac1.shape == (5, 1, 1)  # alpha
    assert np.allclose(jac1[-1], sm.order1["alpha"].F0)

    assert jac2.shape == (5, 2, 1)  # alpha, T2
    assert np.allclose(jac2[-1, 0], sm.order1["alpha"].F0)
    assert np.allclose(jac2[-1, 1], sm.order1["T2"].F0)

    assert jac3.shape == (5, 2, 1)  # magnitude, alpha
    assert np.allclose(jac3[-1, 0], sm.F0)
    assert np.allclose(jac3[-1, 1], sm.order1["alpha"].F0)


def test_hessian_class():
    # order2 probe
    necho = 5
    rf = operators.T(15, 90, order2="alpha", name="T")
    rlx = operators.E(5, 1e3, 30, order2="T2", name="E")
    grd = operators.S(1, name="S")
    adc = operators.ADC

    seq = [rf, rlx, grd, adc] * necho

    probes = [
        diff.Hessian("alpha"),
        diff.Hessian(["alpha", "T2"]),
        diff.Hessian(["magnitude", "alpha"], "T2"),
    ]

    sm = statematrix.StateMatrix()
    tmp = []
    for op in seq:
        sm = op(sm, inplace=False)
        if isinstance(op, operators.Probe):
            tmp.append(probes[0].acquire(sm))

    hes1, hes2, hes3 = functions.simulate(seq, probe=probes)

    assert hes1.shape == (necho, 1, 1, 1)  # alpha
    assert np.allclose(hes1[-1, 0, 0], sm.order2[("alpha", "alpha")].F0)

    assert hes2.shape == (necho, 2, 2, 1)  # alpha, T2
    assert np.allclose(hes2[-1, 0, 0], sm.order2[("alpha", "alpha")].F0)
    assert np.allclose(hes2[-1, 0, 1], sm.order2[("T2", "alpha")].F0)
    assert np.allclose(hes2[-1, 1, 0], sm.order2[("alpha", "T2")].F0)
    assert np.allclose(hes2[-1, 1, 1], sm.order2[("T2", "T2")].F0)

    assert hes3.shape == (necho, 2, 1, 1)  # (magnitude, alpha) x T2
    assert np.allclose(hes3[-1, 0, 0], sm.order1["T2"].F0)  # magnitude
    assert np.allclose(hes3[-1, 1, 0], sm.order2[("T2", "alpha")].F0)

    #
    # vary alpha
    params = [f"alpha_{i:02d}" for i in range(necho)]
    alphas = np.sin(np.arange(necho) * np.pi * 1.3 / necho - np.pi / 2) * 5 + 35

    rfs = lambda alpha, param: operators.T(
        alpha, 90, order1={param: "alpha"}, order2=[("T2", param)]
    )
    rlx = operators.E(5, 1e3, 30, order1="T2", order2="T2")
    grd = operators.S(1)
    adc = operators.ADC

    seq = [[rfs(alphas[i], params[i]), rlx, grd, adc] for i in range(necho)]
    Jacobian = diff.Jacobian(["magnitude", "T2"])
    Hessian = diff.Hessian(["magnitude", "T2"], params)
    jac, hes = functions.simulate(seq, probe=[Jacobian, Hessian])

    # finite diffs
    da = np.random.uniform(-1, 1, necho)
    seq_d = [
        [rfs(alphas[i] + 1e-8 * da[i], params[i]), rlx, grd, adc] for i in range(necho)
    ]
    jac_d = functions.simulate(seq_d, probe=Jacobian)
    assert np.allclose(hes[..., 0] @ da, (jac_d - jac)[..., 0] * 1e8, atol=1e-6)


def test_partials_pruner_class():
    necho = 50

    rfs = {i: operators.T(5, i * (i + 1) / 2, name=f"rf{i:02}") for i in range(necho)}
    rfs[0] = operators.T(5, 0, order2="alpha", name="rf00")
    rlx = operators.E(5, 50, 5, order1="T2")
    grd = operators.S(1)
    adc = operators.ADC

    seq = [[rfs[i], rlx, adc, rlx, grd] for i in range(necho)]
    probe = [diff.Jacobian(["T2", "alpha"]), diff.Hessian("alpha")]

    # no pruning
    jac1, hes1 = functions.simulate(seq, probe=probe)
    assert jac1.shape == (necho, 2, 1)

    assert not np.isclose(jac1[0, 0], 0)
    assert not np.isclose(jac1[0, 1], 0)
    assert not np.isclose(hes1[0], 0)

    assert not np.isclose(jac1[-1, 0], 0)
    assert np.isclose(jac1[-1, 1], 0)  # derivative vanished
    assert np.isclose(hes1[-1], 0)  # derivative vanished
    nonzero1 = np.flatnonzero(jac1[:, 1])

    # with pruning
    pruner = diff.PartialsPruner(condition=1e-5, variables=["alpha"])
    jac2, hes2 = functions.simulate(seq, probe=probe, callback=pruner)
    nonzero2 = np.flatnonzero(jac2[:, 1])
    assert nonzero2.max() < nonzero1.max()

    assert np.allclose(jac1, jac2, atol=1e-6)
    assert np.allclose(jac1[nonzero2], jac2[nonzero2])

    assert np.allclose(hes1, hes2, atol=1e-6)
