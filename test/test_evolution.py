import numpy as np
import pytest
from epgpy import evolution, transition, statematrix

StateMatrix = statematrix.StateMatrix


def test_E_class():
    sm0 = StateMatrix()
    sm1 = transition.T(90, 90)(sm0)

    sm = evolution.E(10, 1e10, 1e10)(sm1)
    assert np.allclose(sm.states, [[[1, 1, 0]]])

    sm = evolution.E(10, 1e10, 1e-10)(sm1)
    assert np.allclose(sm.states, [[[0, 0, 0]]])

    sm = evolution.E(10, 1e-10, 1e-10)(sm1)
    assert np.allclose(sm.states, [[[0, 0, 1]]])

    sm = evolution.E(10, 1e10, 1e10, 0.025)(sm1)
    assert np.allclose(sm.states, [[[1j, -1j, 0]]])

    # multi-dim
    sm = evolution.E(10, 1e10, [1e10, 1e-10])(sm1)
    assert np.allclose(sm.states, [[[1, 1, 0]], [[0, 0, 0]]])

    sm = evolution.E(10, [1e10, 1e-10], 1e10)(sm1)
    assert np.allclose(sm.states, [[[1, 1, 0]], [[1, 1, 1]]])

    sm = evolution.E(10, 1e10, 1e10, [0.025, 0.05])(sm1)
    assert np.allclose(sm.states, [[[1j, -1j, 0]], [[-1, -1, 0]]])

    with pytest.raises(ValueError):
        # incompatible shapes
        evolution.E(10, [1000] * 3, [100] * 2)

    with pytest.raises(ValueError):
        # incompatible shapes
        evolution.E(10, [1000] * 3, 100, [0.025, 0.05])

    # n-dimensional tau
    sm = evolution.E([0, 10], 1e-10, 1e-10)(sm1)
    assert np.allclose(sm.states, [[[1, 1, 0]], [[0, 0, 1]]])

    with pytest.raises(ValueError):
        # incompatible shapes
        evolution.E([10] * 2, [1000] * 3, 100)


def test_P_class():
    """P (only precession)"""

    sm0 = StateMatrix()
    sm1 = transition.T(90, 90)(sm0)

    sm = evolution.P(1, 0.25)(sm1)
    assert np.allclose(sm.states, [[[1j, -1j, 0]]])

    # multi-dim
    sm = evolution.P(1, [0.25, 0.5])(sm1)
    assert np.allclose(sm.states, [[[1j, -1j, 0]], [[-1, -1, 0]]])

    sm0 = StateMatrix(init=[1, 1, 0], shape=(2, 3))
    sm = evolution.P(1, [0.25, 0.5])(sm0)
    assert sm.shape == (2, 3)
    assert np.allclose(sm.states[0], [1j, -1j, 0])
    assert np.allclose(sm.states[1], [-1, -1, 0])


def test_R_class():
    R = evolution.R
    E = evolution.E
    P = evolution.P
    T = transition.T

    op_r = R(1 / 3)
    op_e = E(10, 1e10, 30)
    assert np.allclose(op_r.rL, 0)
    assert op_r.arr0 is None
    assert np.allclose(op_r.arr, op_e.arr)

    op_r = R(1 / 3 + 2j * np.pi, 1 / 10)
    op_e = E(10, 100, 30, g=0.1)
    assert op_r.arr0 is None  # no recovery
    assert np.allclose(op_r.arr, op_e.arr)

    op_r = R(1 / 3 + 2j * np.pi, 1 / 10, r0=1 / 10)
    op_e = E(10, 100, 30, g=0.1)
    assert np.allclose(op_r.arr, op_e.arr)
    assert np.allclose(op_r.arr0, op_e.arr0)

    op_r = R(2j * np.pi)
    op_p = P(10, 0.1)
    assert np.allclose(op_r.rL, 0)
    assert op_r.arr0 is None
    assert np.allclose(op_r.arr, op_p.arr)

    sm0 = StateMatrix()
    op_t = T([90, 0], 90)
    op_r = R(rL=[0, 1 / 10])

    sm1 = (op_r * op_t)(sm0)
    assert np.allclose(sm1.states[0], op_t(sm0).states[0])
    assert np.allclose(sm1.states[1], op_r(sm0).states[1])
    sm2 = (op_r @ op_t)(sm0)
    assert np.allclose(sm1.states, sm2.states)


def test_R_diff():
    dr = 1e-8
    R = evolution.R
    op = R(1/3, 1, r0=2, order1=True, order2=True)
    assert op.parameters_order1 == {'rT', 'rL', 'r0'}
    assert op.parameters_order2 == {('rT', 'rT'), ('rL', 'rL'), ('r0', 'r0')}
    
    sm0 = StateMatrix([1, 1, 0.5])
    sm1 = op(sm0)

    assert set(sm1.order1) == {'rT', 'rL', 'r0'}
    assert set(sm1.order2) == {('rT', 'rT'), ('rL', 'rL'), ('r0', 'r0')}

    # finite diffs
    # order1
    op_rT = R(1/3 + dr, 1, r0=2, order1=True)
    sm1_rT = op_rT(sm0)
    assert np.allclose((sm1_rT.states - sm1.states) * 1e8, sm1.order1['rT'].states)

    op_rL = R(1/3, 1 + dr, r0=2, order1=True)
    sm1_rL = op_rL(sm0)
    assert np.allclose((sm1_rL.states - sm1.states) * 1e8, sm1.order1['rL'].states)

    op_r0 = R(1/3, 1, r0=2 + dr, order1=True)
    sm1_r0 = op_r0(sm0)
    assert np.allclose((sm1_r0.states - sm1.states) * 1e8, sm1.order1['r0'].states)

    # order2
    assert np.allclose((sm1_rL.order1['rL'].states - sm1.order1['rL'].states) * 1e8, sm1.order2[('rL', 'rL')])
    assert np.allclose((sm1_rT.order1['rT'].states - sm1.order1['rT'].states) * 1e8, sm1.order2[('rT', 'rT')])
    assert np.allclose((sm1_r0.order1['r0'].states - sm1.order1['r0'].states) * 1e8, sm1.order2[('r0', 'r0')])


def test_P_diff():
    dt = 1e-8
    P = evolution.P
    op = P(10, 0.1, order1=True, order2=True)
    assert op.parameters_order1 == {'tau', 'g'}
    assert op.parameters_order2 == {('tau', 'tau'), ('g', 'g'), ('g', 'tau')}
    
    sm0 = StateMatrix([1, 1, 0.5])
    sm1 = op(sm0)

    assert set(sm1.order1) == op.parameters_order1
    assert {tuple(sorted(pair)) for pair in sm1.order2} == op.parameters_order2

    # finite diffs
    # order1
    op_tau = P(10 + dt, 0.1, order1=True)
    sm1_tau = op_tau(sm0)
    assert np.allclose((sm1_tau.states - sm1.states) * 1e8, sm1.order1['tau'].states)

    op_g = P(10, 0.1 + dt, order1=True)
    sm1_g = op_g(sm0)
    assert np.allclose((sm1_g.states - sm1.states) * 1e8, sm1.order1['g'].states)

    # order2
    assert np.allclose(
        (sm1_tau.order1['tau'].states - sm1.order1['tau'].states) * 1e8, 
        sm1.order2[('tau', 'tau')].states,
    )

    assert np.allclose(
        (sm1_g.order1['g'].states - sm1.order1['g'].states) * 1e8, 
        sm1.order2[('g', 'g')].states,
    )

    assert np.allclose(
        (sm1_tau.order1['g'].states - sm1.order1['g'].states) * 1e8, 
        sm1.order2[('g', 'tau')].states,
    )
    assert np.allclose(
        (sm1_g.order1['tau'].states - sm1.order1['tau'].states) * 1e8, 
        sm1.order2[('g', 'tau')].states,
    )


def test_E_diff():
    dt = 1e-8
    E = evolution.E
    op = E(10, 20, 30, 0.1, order1=True, order2=True)
    assert op.parameters_order1 == {'tau', 'T1', 'T2', 'g'}
    assert op.parameters_order2 == {
        ('tau', 'tau'), ('T1', 'T1'), ('T2', 'T2'), ('g', 'g'),
        ('T1', 'tau'), ('T2', 'tau'), ('g', 'tau'), ('T2', 'g'),
        }
    
    sm0 = StateMatrix([1, 1, 0.5])
    sm1 = op(sm0)

    assert set(sm1.order1) == op.parameters_order1
    assert {tuple(sorted(pair)) for pair in sm1.order2} ==  op.parameters_order2

    # finite diffs
    # order1
    op_tau = E(10 + dt, 20, 30, 0.1, order1=True)
    sm1_tau = op_tau(sm0)
    assert np.allclose((sm1_tau.states - sm1.states) * 1e8, sm1.order1['tau'].states)

    op_T1 = E(10, 20 + dt, 30, 0.1, order1=True)
    sm1_T1 = op_T1(sm0)
    assert np.allclose((sm1_T1.states - sm1.states) * 1e8, sm1.order1['T1'].states)

    op_T2 = E(10, 20, 30 + dt, 0.1, order1=True)
    sm1_T2 = op_T2(sm0)
    assert np.allclose((sm1_T2.states - sm1.states) * 1e8, sm1.order1['T2'].states)

    op_g = E(10, 20, 30, 0.1 + dt, order1=True)
    sm1_g = op_g(sm0)
    assert np.allclose((sm1_g.states - sm1.states) * 1e8, sm1.order1['g'].states)

    # order2
    assert np.allclose(
        (sm1_tau.order1['tau'].states - sm1.order1['tau']) * 1e8, 
        sm1.order2[('tau', 'tau')].states,
    )
    assert np.allclose(
        (sm1_T1.order1['T1'].states - sm1.order1['T1']) * 1e8, 
        sm1.order2[('T1', 'T1')].states,
    )
    assert np.allclose(
        (sm1_T2.order1['T2'].states - sm1.order1['T2']) * 1e8, 
        sm1.order2[('T2', 'T2')].states,
    )
    assert np.allclose(
        (sm1_g.order1['g'].states - sm1.order1['g']) * 1e8, 
        sm1.order2[('g', 'g')].states,
    )
    assert np.allclose(
        (sm1_tau.order1['T1'].states - sm1.order1['T1']) * 1e8, 
        sm1.order2[('T1', 'tau')].states,
    )
    assert np.allclose(
        (sm1_tau.order1['T2'].states - sm1.order1['T2']) * 1e8, 
        sm1.order2[('T2', 'tau')].states,
    )
    assert np.allclose(
        (sm1_tau.order1['g'].states - sm1.order1['g']) * 1e8, 
        sm1.order2[('g', 'tau')].states,
    )
    assert np.allclose(
        (sm1_T2.order1['g'].states - sm1.order1['g']) * 1e8, 
        sm1.order2[('T2', 'g')].states,
    )