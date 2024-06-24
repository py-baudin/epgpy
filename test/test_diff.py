""" unittest for epgpy.diff """
import pytest
import numpy as np
from epgpy import statematrix, diff, operators, functions


def test_parse_partials():

    class Op(diff.DiffOperator):
        parameters = ["x", "y"]
        def _apply(self, sm): pass
        def _derive1(self, *args): pass
        def _derive2(self, sm, params): pass
        def __repr__(self): return "op"

    # dummy operator
    op = Op()

    coeffs1, coeffs2 = op._parse_partials()
    assert not coeffs1 and not coeffs2

    coeffs1, coeffs2 = op._parse_partials(order1=True)
    assert not coeffs2
    assert coeffs1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}

    coeffs1, coeffs2 = op._parse_partials(order1=True, isolated=False)
    assert coeffs1 == {'x': {"x": 1}, 'y': {"y": 1}}

    coeffs1, coeffs2 = op._parse_partials(order1="x")
    assert not coeffs2
    assert coeffs1 == {(op, "x"): {"x": 1}}

    coeffs1, coeffs2 = op._parse_partials(order1="x", isolated=False)
    assert coeffs1 == {'x': {"x": 1}}

    coeffs1, coeffs2 = op._parse_partials(order1={"z": {"x": 2, "y": 3}})
    assert not coeffs2
    assert coeffs1 == {"z": {"x": 2, "y": 3}}

    with pytest.raises(ValueError):
        op._parse_partials(order1="unknown")

    with pytest.raises(ValueError):
        op._parse_partials(order1={"z": {"unknown": 1}})

    coeffs1, coeffs2 = op._parse_partials(order2=True)
    assert coeffs1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}
    assert coeffs2 == {
        ((op, "x"), (op, "x")): {},
        ((op, "x"), (op, "y")): {},
        ((op, "y"), (op, "y")): {},
    }

    coeffs1, coeffs2 = op._parse_partials(order2="x")
    assert coeffs1 == {(op, "x"): {"x": 1}}
    assert coeffs2 == {((op, "x"), (op, "x")): {}}

    coeffs1, coeffs2 = op._parse_partials(order2="x", isolated=False)
    assert coeffs1 == {'x': {"x": 1}}
    assert coeffs2 == {('x', 'x'): {}}

    coeffs1, coeffs2 = op._parse_partials(
        order1={"foo": {"x": 1, "y": 2}, "bar": {"x": 1, "y": 2}},
        order2={("foo", "bar"): {"x": 2, "y": 3}},
    )
    assert coeffs1 == {"foo": {"x": 1, "y": 2}, "bar": {"x": 1, "y": 2}}
    assert coeffs2 == {("foo", "bar"): {"x": 2, "y": 3}}

    with pytest.raises(ValueError):
        op._parse_partials(order2={("foo", "bar"): {"x": 2, "y": 3}})


def test_order12():
    """ Test order 1 and 2 partials"""

    class Op(diff.DiffOperator):
        parameters = ["x", "y"]
        def _apply(self, sm): return sm
        def _derive1(self, sm, param):
            op1 = {'x': 2, 'y': 3}[param]
            return op1 * sm
        def _derive2(self, sm, pair): 
            pair = tuple(sorted(pair))
            op2 = {("x", "x"): 4, ("x", "y"): 5, ("y", "y"): 6}[pair]
            return op2 * sm

    # order 1
    op = Op(order1=True)
    # assert op.coeffs1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}}
    assert op.coeffs1 == {'x': {'x': 1}, 'y': {"y": 1}}
    assert not op.coeffs2
    assert op.parameters_order1 == {"x", "y"}

    # nd state matrix
    sm0 = statematrix.StateMatrix([[[0, 0, 1]], [[1, 1, 0]], [[1 + 1j, 1 - 1j, 0]]])
    # apply operator
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states) # sm is unchanged
    # assert {(op, "x"), (op, "y")} == set(sm.order1) # order1 partials were computed
    assert {'x', 'y'} == set(sm.order1) # order1 partials were computed
    # assert np.allclose(sm.order1[(op, "x")], 2 * sm0.states)
    # assert np.allclose(sm.order1[(op, "y")], 3 * sm0.states)
    assert np.allclose(sm.order1["x"], 2 * sm0.states)
    assert np.allclose(sm.order1["y"], 3 * sm0.states)

    # order 2
    op = Op(order2=True)
    # order 1 is filled by default
    # assert op.coeffs1 == {(op, "x"): {"x": 1}, (op, "y"): {"y": 1}} 
    assert op.coeffs1 == {"x": {"x": 1}, "y": {"y": 1}} 
    assert op.coeffs2 == {
        # ((op, "x"), (op, "x")): {},
        # ((op, "x"), (op, "y")): {},
        # ((op, "y"), (op, "y")): {},
        ("x", "x"): {},
        ("x", "y"): {},
        ("y", "y"): {},
    }
    assert op.parameters_order1 == {"x", "y"}
    assert op.parameters_order2 == {('x', 'y'), ('x', 'x'), ('y', 'y')} # {"x", "y"}
    # apply operator
    sm = op(sm0)
    assert np.allclose(sm.states, sm0.states) # sm is unchanged
    # order1 partials were computed
    # assert {(op, "x"), (op, "y")} == set(sm.order1)
    assert {"x", "y"} == set(sm.order1)
    # order2 partials were computed
    # assert {((op, "x"), (op, "x")), ((op, "x"), (op, "y")), ((op, "y"), (op, "x")), ((op, "y"), (op, "y"))} == set(sm.order2)
    assert {("x", "x"), ("x", "y"), ("y", "x"), ("y", "y")} == set(sm.order2)
    # assert np.allclose(sm.order1[(op, "x")], 2 * sm0.states)
    # assert np.allclose(sm.order1[(op, "y")], 3 * sm0.states)
    assert np.allclose(sm.order1["x"], 2 * sm0.states)
    assert np.allclose(sm.order1["y"], 3 * sm0.states)
    # assert np.allclose(sm.order2[(op, "x"), (op, "x")], 4 * sm0.states)
    # assert np.allclose(sm.order2[(op, "x"), (op, "y")], 5 * sm0.states)
    # assert np.allclose(sm.order2[(op, "y"), (op, "y")], 6 * sm0.states)
    assert np.allclose(sm.order2["x", "x"], 4 * sm0.states)
    assert np.allclose(sm.order2["x", "y"], 5 * sm0.states)
    assert np.allclose(sm.order2["y", "y"], 6 * sm0.states)


    # Arbitrary variable `a`
    op = Op(order1={"a": {"x": 0.1, "y": 0.2}}, order2={("a", "a"): {"x": 0.3}})
    assert op.parameters_order1 == {'x', 'y'}
    # assert op.parameters_order2 == {('x', 'y')} # {'x', 'y'}
    assert op.parameters_order2 == {('x', 'y'), ('x', 'x'), ('y', 'y')} 
    # apply operator
    sm = op(sm0)
    assert np.allclose(sm.order1["a"], (0.1 * 2 + 0.2 * 3) * sm0.states)
    assert np.allclose(
        sm.order2[("a", "a")],
        (4 * 0.1 ** 2 + 2 * 5 * 0.1 * 0.2 + 6 * 0.2 ** 2 + 0.3 * 2) * sm0.states,
    )


def test_diff_chain():
    """ chain multiple differentiable operators """
    class Op(diff.DiffOperator):
        parameters = ["x", "y"]
        def _apply(self, sm): return sm
        def _derive1(self, sm, param):
            op1 = {'x': 2, 'y': 3}[param]
            return op1 * sm
        def _derive2(self, sm, pair): 
            pair = tuple(sorted(pair))
            op2 = {("x", "x"): 4, ("x", "y"): 5, ("y", "y"): 6}[pair]
            return op2 * sm
        
    # order 1
    op1 = Op(order1={'a': {'x': 0.1}, 'b': {'y': 0.2}})
    op2 = Op(order1={'a': {'y': 0.3}, 'b': {'x': 0.4}})
    sm0 = statematrix.StateMatrix()
    sm1 = op2(op1(sm0))
    assert set(sm1.order1) == {'a', 'b'}
    # op2/d1 * op1 + op2 * op1/d1
    assert np.allclose(sm1.order1['a'].states, (0.1 * 2 + 0.3 * 3) * sm0.states)
    assert np.allclose(sm1.order1['b'].states, (0.2 * 3 + 0.4 * 2) * sm0.states)

    # order 2
    op1 = Op(order1={'a': {'x': 0.1}, 'b': {'y': 0.2}}, order2={('a', 'b'): {}})
    op2 = Op(order1={'a': {'y': 0.3}, 'b': {'x': 0.4}})
    sm0 = statematrix.StateMatrix()
    sm1 = op1(sm0)
     # op1/d1d2
    assert set(sm1.order2) == {('a', 'b'), ('b', 'a')}
    assert np.allclose(sm1.order2[('a', 'b')].states, (0.1 * 0.2 * 5) * sm0.states)
    assert np.allclose(sm1.order2[('a', 'b')].states, sm1.order2[('b', 'a')].states)
    sm2 = op2(sm1)
    # (op2/d1d2 * op1) + (op2/d1 * op1/d2) + (op2/d2 * op1/d1) + (op2 * op1/d1d2)
    assert np.allclose(sm2.order2[('a', 'a')].states, (2 * 0.3 * 3 * 0.1 * 2) * sm0.states)
    assert np.allclose(
        sm2.order2[('a', 'b')].states, 
        ((0.3 * 3 * 0.2 * 3) + (0.4 * 2 * 0.1 * 2) + (0.1 * 0.2 * 5)) * sm0.states,
    )
    assert np.allclose(sm2.order2[('a', 'b')].states, sm2.order2[('b', 'a')])
    assert np.allclose(sm2.order2[('b', 'b')].states, (2 * 0.4 * 2 * 0.2 * 3) * sm0.states)


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
        sm_T2 = rlx_T2(sm_T2) if op.name == 'relax' else op(sm_T2)
        sm_alpha = ref_alpha(sm_alpha) if op.name == 'ref' else op(sm_alpha)
        states.append(sm)

    # check derivatives
    assert set(sm.order1) == {'T2', 'alpha'}

    # compare to finite difference
    assert np.allclose(
        (sm_T2.states - sm.states) * 1e8, sm.order1['T2'].states, atol=1e-7
    )
    assert np.allclose(
        (sm_alpha.states - sm.states) * 1e8, sm.order1['alpha'].states, atol=1e-7
    )

    # using simulate
    probe = ["F0", diff.Jacobian('T2'), diff.Jacobian('alpha')]
    spinecho = [exc] + [grad, relax, ref, grad, relax, operators.ADC] * necho
    signal, gradT2, gradalpha = functions.simulate(
        spinecho,
        init=[0, 0, 1],
        probe=probe,
    )
    assert np.allclose(signal[-1], sm.F0)
    assert np.allclose(gradT2[-1], sm.order1['T2'].F0)
    assert np.allclose(gradalpha[-1], sm.order1['alpha'].F0)


def test_diff2_ssfp():
    """test 2nd order differentials"""
    alpha, phi = 20, 90
    tau, T1, T2 = 5, 1e3, 30

    rf = operators.T(alpha, phi, name="pulse", order1=True, order2="alpha")
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
    for i, op in enumerate(seq):
        sm = op(sm)
        sm_T2 = rlx_T2(sm_T2) if op.name == "relax" else op(sm_T2)
        sm_alpha = rf_alpha(sm_alpha) if op.name == "pulse" else op(sm_alpha)

    assert set(sm.order1) == {'alpha', 'phi', 'T2'}
    assert set(sm.order2) == {('alpha', 'alpha'), ('T2', 'T2'), ('alpha', 'T2'), ('T2', 'alpha')}

    # finite difference

    # order1
    assert np.allclose((sm_alpha.states - sm.states) * 1e8, sm.order1['alpha'].states, atol=1e-7)
    assert np.allclose((sm_T2.states - sm.states) * 1e8, sm.order1['T2'].states, atol=1e-7)

    # order2
    assert np.allclose(
        (sm_alpha.order1['alpha'].states - sm.order1['alpha'].states) * 1e8, 
        sm.order2[('alpha', 'alpha')].states,
    )

    assert np.allclose(
        (sm_alpha.order1['T2'].states - sm.order1['T2'].states) * 1e8, 
        sm.order2[('alpha', 'T2')].states,
    )

    assert np.allclose(
        (sm_T2.order1['alpha'].states - sm.order1['alpha'].states) * 1e8, 
        sm.order2[('alpha', 'T2')].states,
    )

    assert np.allclose(
        (sm_T2.order1['T2'].states - sm.order1['T2'].states) * 1e8, 
        sm.order2[('T2', 'T2')].states,
    )

    # using simulate
    probe = [
        lambda sm: sm.order2[("alpha", "alpha")].F0,
        lambda sm: sm.order2[("alpha", "T2")].F0,
        lambda sm: sm.order2[("T2", "T2")].F0,
    ]
    order2 = functions.simulate(
        seq + [operators.ADC],
        probe=probe,
        squeeze=False,  # tmp
    )
    assert np.isclose(order2[0], sm.order2[('alpha', 'alpha')].F0)
    assert np.isclose(order2[1], sm.order2[('alpha', 'T2')].F0)
    assert np.isclose(order2[2], sm.order2[('T2', 'T2')].F0)


# def test_A_combine():
#     """test combining B operators"""
#     E = diff.E(5, 1e2, 5e2, g=1e-1, order1="T2", name="E")
#     Ta = diff.T(15, 90, order1={"b1": {"alpha": 15}, "phi": {"phi": 1}}, name="Ta")
#     Tb = diff.T(20, 90, order1={"b1": {"alpha": 20}, "phi": {"phi": 1}}, name="Tb")

#     assert E.diff1 == {"T2"}
#     assert Ta.diff1 == {"alpha", "phi"}
#     assert Tb.diff1 == {"alpha", "phi"}

#     # combine operators
#     ops = [Ta, E, Tb, E, Ta, E, Tb, E]
#     combined = diff.A.combine(ops)
#     assert combined.diff1 == {
#         (E, "T2"),
#         "b1",
#         "phi",
#     }

#     # simulate separately
#     sm0 = core.StateMatrix([1, 1, 0])
#     sm1 = sm0
#     for i, op in enumerate(ops):
#         sm1 = op(sm1)

#     sm2 = core.StateMatrix([1, 1, 0])
#     sm2 = combined(sm2)
#     # assert set(sm2.order1) == combined.diff1
#     assert set(sm2.order1) == set(combined.coeffs1)

#     assert np.allclose(sm1.states, sm2.states)
#     for partial in combined.coeffs1:
#         assert np.allclose(sm1.order1[partial].states, sm2.order1[partial])

#     # 2nd order combine
#     E = diff.E(5, 1e2, 5e2, g=1e-1, order2="T2", name="E")
#     Ta = diff.T(15, 90, order2=True, name="Ta")
#     Tb = diff.T(20, 90, order2=True, name="Tb")

#     assert E.diff2 == {"T2"}
#     assert E.auto_cross_derivatives
#     assert Ta.diff2 == {"alpha", "phi"}
#     assert Ta.auto_cross_derivatives
#     assert Tb.diff2 == {"alpha", "phi"}
#     assert Tb.auto_cross_derivatives

#     # combine operators
#     ops = [Ta, E, Tb, E, Ta, E, Tb, E]
#     combined = diff.A.combine(ops)
#     assert combined.diff2 == {
#         (E, "T2"),
#         (Ta, "alpha"),
#         (Ta, "phi"),
#         (Tb, "alpha"),
#         (Tb, "phi"),
#     }

#     # simulate separately
#     sm0 = core.StateMatrix([1, 1, 0])
#     sm1 = sm0
#     for i, op in enumerate(ops):
#         sm1 = op(sm1)

#     sm2 = core.StateMatrix([1, 1, 0])
#     sm2 = combined(sm2)

#     assert set(sm2.order1) == set(sm1.order1)
#     for var in sm1.order1:
#         assert np.allclose(sm1.order1[var].states, sm2.order1[var].states)

#     assert set(sm2.order2) == set(sm1.order2)
#     for pair in sm1.order2:
#         assert np.allclose(sm1.order2[pair].states, sm2.order2[pair].states)


# def test_combine_2():
#     E = diff.E(10, 1e3, 35, order2=["T1", "T2"], name="E")
#     T = diff.T(150, 0, order2=["alpha"], name="T")

#     sm0 = core.StateMatrix([1, 1, 0])
#     sm1 = E(T(E(sm0)))

#     combined = diff.A.combine([E, T, E])
#     sm2 = combined(sm0)

#     assert np.allclose(sm1.states, sm2.states)
#     assert set(sm1.order1) == set(sm2.order1)
#     for var in sm1.order1:
#         assert np.allclose(sm1.order1[var], sm2.order1[var])
#     assert set(sm1.order2) == set(sm2.order2)
#     pairs = list(sm1.order2)
#     for var in sm1.order2:
#         assert np.allclose(sm1.order2[var], sm2.order2[var])

#     # repeat combined
#     sm1 = E(T(E(sm0)))
#     smx = combined(sm0)

#     _combined = diff.A.combine([T, E])
#     combined2 = diff.A.combine([E, _combined])
#     sm2 = combined2(sm0)

#     assert np.allclose(sm1.states, sm2.states)
#     assert set(sm1.order1) == set(sm2.order1)
#     for var in sm1.order1:
#         assert np.allclose(sm1.order1[var], sm2.order1[var])
#     assert set(sm1.order2) == set(sm2.order2)
#     pairs = list(sm1.order2)
#     for var in sm1.order2:
#         assert np.allclose(sm1.order2[var], sm2.order2[var])


def test_jacobian_class():
    # jacobian probe
    necho = 5
    rf = diff.T(15, 90, order1=["alpha"])
    relax = diff.E(5, 1e3, 30, order1=["T2"])
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
    assert np.allclose(jac1[-1], sm.order1[(rf, "alpha")].F0)

    assert jac2.shape == (5, 2, 1)  # alpha, T2
    assert np.allclose(jac2[-1, 0], sm.order1[(rf, "alpha")].F0)
    assert np.allclose(jac2[-1, 1], sm.order1[(relax, "T2")].F0)

    assert jac3.shape == (5, 2, 1)  # magnitude, alpha
    assert np.allclose(jac3[-1, 0], sm.F0)
    assert np.allclose(jac3[-1, 1], sm.order1[(rf, "alpha")].F0)


def test_hessian_class():
    # order2 probe
    necho = 5
    rf = diff.T(15, 90, order2=["alpha"], name="T")
    relax = diff.E(5, 1e3, 30, order2=["T2"], name="E")
    shift = diff.S(1, name="S")
    adc = core.ADC

    seq = [rf, relax, shift, adc] * necho

    probes = [
        diff.order2((rf, "alpha")),
        diff.order2([(rf, "alpha"), (relax, "T2")]),
        diff.order2(["magnitude", (rf, "alpha")], (relax, "T2")),
    ]

    sm = core.StateMatrix()
    tmp = []
    for op in seq:
        sm = op(sm, inplace=False)
        if isinstance(op, core.Probe):
            tmp.append(probes[0].acquire(sm))

    hes1, hes2, hes3 = core.simulate(seq, probe=probes, squeeze=False)

    assert hes1.shape == (necho, 1, 1, 1)  # alpha
    assert np.allclose(hes1[-1, 0, 0], sm.order2[((rf, "alpha"), (rf, "alpha"))].F0)

    assert hes2.shape == (necho, 2, 2, 1)  # alpha, T2
    assert np.allclose(hes2[-1, 0, 0], sm.order2[((rf, "alpha"), (rf, "alpha"))].F0)
    assert np.allclose(hes2[-1, 0, 1], sm.order2[((relax, "T2"), (rf, "alpha"))].F0)
    assert np.allclose(hes2[-1, 1, 0], sm.order2[((rf, "alpha"), (relax, "T2"))].F0)
    assert np.allclose(hes2[-1, 1, 1], sm.order2[((relax, "T2"), (relax, "T2"))].F0)

    assert hes3.shape == (necho, 2, 1, 1)  # (magnitude, alpha) x T2
    assert np.allclose(hes3[-1, 0, 0], sm.order1[(relax, "T2")].F0)  # magnitude
    assert np.allclose(hes3[-1, 1, 0], sm.order2[((rf, "alpha"), (relax, "T2"))].F0)


def test_partial_hessian():
    # partial order2
    necho = 2
    rf = diff.T(
        15,
        90,
        order1={"alpha": {"alpha": 1}},
        order2={("alpha", "T2"): {}, ("alpha", "T1"): {}},
        name="rf1",
    )
    relax = diff.E(
        5,
        1e3,
        30,
        order1={"T2": {"T2": 1}, "T1": {"T1": 1}},
        order2={("alpha", "T2"): {}, ("alpha", "T1"): {}},
        name="relax1",
    )
    shift = diff.S(1, name="shift1")
    adc = core.ADC

    seq = [rf, relax, shift, adc] * necho

    sm = core.StateMatrix()
    for op in seq:
        sm = op(sm)

    assert set(sm.order2) == {
        ("alpha", "T2"),
        ("alpha", "T1"),
        ("T1", "alpha"),
        ("T2", "alpha"),
    }

    # full order2
    rf_ = diff.T(15, 90, order2=True, name="rf2")
    relax_ = diff.E(5, 1e3, 30, order2=True, name="relax2")
    shift = diff.S(1, name="shift2")
    seq_ = [rf_, relax_, shift, adc] * necho
    sm2 = core.StateMatrix()
    for op in seq_:
        sm2 = op(sm2)
    assert np.allclose(
        sm2.order2[(rf_, "alpha"), (relax_, "T2")], sm.order2[("alpha", "T2")]
    )

    # finite diff
    rf_ = diff.T(15 + 1e-8, 90)
    seq_ = [rf_, relax, shift, adc] * necho
    sm_ = core.StateMatrix()
    for op in seq_:
        sm_ = op(sm_)
    fdiff = (sm_.order1["T2"].F0 - sm.order1["T2"].F0) * 1e8
    assert np.allclose(fdiff, sm.order2[("alpha", "T2")].F0)


def test_pruning():
    necho = 50

    shift = diff.S(1)
    relax = diff.E(5, 50, 5, order1=["T2"])
    adc = core.ADC

    def rf(i):
        order2 = "alpha" if i == 0 else None
        return diff.T(5, i ** 2, order2=order2, name=f"T_{i:02}")

    seq = [[rf(i), relax, adc, relax, shift] for i in range(necho)]
    variables = [(relax, "T2"), (seq[0][0], "alpha")]

    # no pruning
    probe = [diff.Jacobian(variables), diff.order2(variables[1])]
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
