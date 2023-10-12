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
    assert op_r.coeff0 is None
    assert np.allclose(op_r.coeff, op_e.coeff)

    op_r = R(1 / 3 + 2j * np.pi, 1 / 10)
    op_e = E(10, 100, 30, g=0.1)
    assert op_r.coeff0 is None  # no recovery
    assert np.allclose(op_r.coeff, op_e.coeff)

    op_r = R(1 / 3 + 2j * np.pi, 1 / 10, r0=1 / 10)
    op_e = E(10, 100, 30, g=0.1)
    assert np.allclose(op_r.coeff, op_e.coeff)
    assert np.allclose(op_r.coeff0, op_e.coeff0)

    op_r = R(2j * np.pi)
    op_p = P(10, 0.1)
    assert np.allclose(op_r.rL, 0)
    assert op_r.coeff0 is None
    assert np.allclose(op_r.coeff, op_p.coeff)

    sm0 = StateMatrix()
    op_t = T([90, 0], 90)
    op_r = R(rL=[0, 1 / 10])

    sm1 = (op_r * op_t)(sm0)
    assert np.allclose(sm1.states[0], op_t(sm0).states[0])
    assert np.allclose(sm1.states[1], op_r(sm0).states[1])
    sm2 = (op_r @ op_t)(sm0)
    assert np.allclose(sm1.states, sm2.states)
