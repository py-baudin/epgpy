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


def Phi_class():
    sm0 = StateMatrix()
    sm1 = StateMatrix([1, 1, 0])
    sm1j = StateMatrix([1j, -1j, 0])
    Phi = transition.Phi

    # T
    assert Phi(90)(sm0).shape == (1,)
    assert np.allclose(Phi(90)(sm).states, [[[0, 0, 1]]])
    assert np.allclose(Phi(90)(sm1).states, [[[1j, -1j, 1]]])
    assert np.allclose(Phi(90)(sm1j).states, [[[-1, -1, 1]]])

    assert np.allclose(Phi([0, 90])(sm1).states, [[[1, 1, 1]], [[1j, -1j, 1]]])
