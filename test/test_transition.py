import numpy as np
import pytest
from epgpy import transition, statematrix

StateMatrix = statematrix.StateMatrix


def test_T_class():
    sm0 = StateMatrix()

    # T
    sm = transition.T(90, 90)(sm0)
    assert np.allclose(sm.states, [[[1, 1, 0]]])

    sm = transition.T(90, 0)(sm0)
    assert np.allclose(sm.states, [[[-1j, 1j, 0]]])

    # multi-dim
    sm = transition.T(90, [90, 0])(sm0)
    assert sm.shape == (2,)
    assert np.allclose(sm.states, [[[1, 1, 0]], [[-1j, 1j, 0]]])

    sm = transition.T(90, [[90, 0]])(sm0)
    assert sm.shape == (1, 2)
    assert np.allclose(sm.states, [[[[1, 1, 0]], [[-1j, 1j, 0]]]])

    sm0 = StateMatrix(shape=(1, 2))
    sm = transition.T(90, 90)(sm0)
    assert np.allclose(sm.states, [[[1, 1, 0]], [[1, 1, 0]]])

    # broadcast states dimension
    op = transition.T([[90, 90]], [0, 90])
    assert op.shape == (2, 2)
    sm = op(sm0)
    assert sm.shape == (2, 2)
    assert np.allclose(sm.states[0], [[[-1j, 1j, 0]], [[-1j, 1j, 0]]])
    assert np.allclose(sm.states[1], [[[1, 1, 0]], [[1, 1, 0]]])

    # broadcast operator dimension
    op = transition.T(90, [[0], [90]])
    sm0 = StateMatrix(shape=(1, 2))
    sm = op(sm0)
    assert sm.shape == (2, 2)

    with pytest.raises(ValueError):
        # incompatible shapes
        transition.T([90] * 2, [90] * 3)

    with pytest.raises(ValueError):
        # incompatible shapes
        transition.T(90, [90] * 3)(StateMatrix(shape=(4,)))

    # alpha = 0
    sm0 = StateMatrix([1, 1, 0])
    sm1 = transition.T(0, 90)(sm0)  # does nothing!
    assert np.allclose(sm0.states, sm1.states)


def test_Phi_class():
    sm0 = StateMatrix()
    sm1 = StateMatrix([1, 1, 0])
    sm1j = StateMatrix([1j, -1j, 0])
    Phi = transition.Phi

    # T
    assert Phi(90)(sm0).shape == (1,)
    assert np.allclose(Phi(90)(sm0).states, [[[0, 0, 1]]])
    assert np.allclose(Phi(90)(sm1).states, [[[1j, -1j, 0]]])
    assert np.allclose(Phi(90)(sm1j).states, [[[-1, -1, 0]]])

    assert np.allclose(Phi([0, 90])(sm1).states, [[[1, 1, 0]], [[1j, -1j, 0]]])


def test_T_diff():
    T = transition.T

    op = T(45, 45, order1=True, order2=True)
    assert op.parameters_order1 == {'alpha', 'phi'}
    assert op.parameters_order2 == {('alpha', 'alpha'), ('phi', 'phi'), ('alpha', 'phi')}

    sm0 = StateMatrix([1, 1, 0.5])
    sm1 = op(sm0)
    assert set(sm1.order1) == op.parameters_order1
    assert {tuple(sorted(pair)) for pair in sm1.order2} == op.parameters_order2

    # finite diff
    da = 1e-8

    # order1
    op_alpha = T(45 + da, 45, order1=True)
    sm1_alpha = op_alpha(sm0)
    assert np.allclose((sm1_alpha.states - sm1.states) / da, sm1.order1['alpha'], atol=1e-6)

    op_phi = T(45, 45 + da, order1=True)
    sm1_phi = op_phi(sm0)
    assert np.allclose((sm1_phi.states - sm1.states) / da, sm1.order1['phi'], atol=1e-6)

    # order2
    assert np.allclose(
        (sm1_alpha.order1['alpha'].states - sm1.order1['alpha'].states) / da, 
        sm1.order2[('alpha', 'alpha')], atol=1e-6,
    )

    assert np.allclose(
        (sm1_phi.order1['phi'].states - sm1.order1['phi'].states) / da, 
        sm1.order2[('phi', 'phi')], atol=1e-6,
    )

    assert np.allclose(
        (sm1_alpha.order1['phi'].states - sm1.order1['phi'].states) / da, 
        sm1.order2[('alpha', 'phi')], atol=1e-6,
    )

def test_Phi_diff():
    Phi = transition.Phi

    op = Phi(45, order1=True, order2=True)
    assert op.parameters_order1 == {'phi'}
    assert op.parameters_order2 == {('phi', 'phi')}

    sm0 = StateMatrix([1, 1, 0.5])
    sm1 = op(sm0)
    assert set(sm1.order1) == op.parameters_order1
    assert set(sm1.order2) == op.parameters_order2

    # finite diff
    da = 1e-8

    # order1
    op_phi = Phi(45 + da, order1=True)
    sm1_phi = op_phi(sm0)
    assert np.allclose((sm1_phi.states - sm1.states) / da, sm1.order1['phi'], atol=1e-6)

    assert np.allclose(
        (sm1_phi.order1['phi'].states - sm1.order1['phi'].states) / da, 
        sm1.order2[('phi', 'phi')], atol=1e-6,
    )
