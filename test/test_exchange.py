import numpy as np
import pytest
from epgpy import exchange, statematrix

StateMatrix = statematrix.StateMatrix


def test_expm():
    # real
    A = np.random.uniform(-1, 1, (3, 3))
    eA = exchange.expm(A)
    ieA = exchange.expm(-A)
    assert np.allclose(eA @ ieA, np.eye(3))

    # complex
    A = np.random.uniform(-1, 1, (3, 3, 2)) @ [1, 1j]
    eA = exchange.expm(A)
    ieA = exchange.expm(-A)
    assert np.allclose(eA @ ieA, np.eye(3))

    # hermician
    A = A + A.T.conj()
    eA = exchange.expm(A)
    ieA = exchange.expm(-A)
    assert np.allclose(eA @ ieA, np.eye(3))


def test_X_class():
    """test exchange operator"""

    # initial state matrices
    sm0 = StateMatrix([1, 1, 0])

    #  no exchange, no relaxation
    op = exchange.X(10, 0)
    sm1 = op(sm0)
    assert np.allclose(sm1.states, sm0.states)  # unchanged

    # no relaxation, some exchange, same initial conditions
    op = exchange.X(10, 0.1)
    sm1 = op(sm0)
    assert np.allclose(sm1.states, sm0.states)  # unchanged

    # no relaxation, fast exchange rate, different initial conditions
    sm0j = StateMatrix([[[1, 1, 0]], [[1j, -1j, 0]]])
    op = exchange.X(10, 1)
    sm1 = op(sm0j)
    assert np.allclose(sm1.states, np.mean(sm0j.states, axis=0))  # mixed

    # pure relaxation,
    op = exchange.X(10, 0, T2=[np.inf, 1e-8])
    sm1 = op(sm0)
    assert np.allclose(sm1.states[0], sm0.states[0])
    assert np.allclose(sm1.states[1], 0)

    # some relaxation, fast exchange rate
    op = exchange.X(10, 10, T2=[np.inf, 1e-8])
    sm1 = op(sm0)
    assert np.allclose(sm1.states, 0)  # all is 0

    # some relaxation, fast exchange rate
    op = exchange.X(10, 1e10, T2=[30, 40])
    sm1 = op(sm0)
    mean_relax = np.exp(-np.mean(op.tau / op.T2))
    assert np.allclose(sm1.states, [mean_relax] * 2 + [0])

    # different densities
    # fast recovery, no echange
    sm0x = StateMatrix([[[1, 1, 0]], [[3, 3, 0]]], density=[1, 3])
    op = exchange.X(10, 0, T1=1e-10, T2=1e-10)
    sm1 = op(sm0x)
    assert np.allclose(sm1.states, [[[0, 0, 1]], [[0, 0, 3]]])

    # fast exchange no recovery
    khi = [[3e10, -1e10], [-3e10, 1e10]]
    op = exchange.X(10, khi)
    sm1 = op(sm0x)
    assert np.allclose(sm1.states, sm0x)

    # non-conserving exchange
    with pytest.raises(RuntimeError):
        khi = [[-10, 0], [10, 0]]
        op = exchange.X(10, khi)
        sm1 = op(sm0j)

    # different axis
    op = exchange.X(10, 0.1, axis=1, T2=[np.inf, 10, 1e-8])
    sm1 = op(sm0)
    assert sm1.shape == (3, 2)
    assert np.allclose(sm1.states[0], sm0.states[0])
    assert np.allclose(sm1.states[2], 0)

    # multiple tau
    sm0_ = StateMatrix([[[1, 1, 0]], [[0.1, 0.1, 0]]])  # different starting points
    op = exchange.X([[1e-10, 1e10]], 1, axis=0)
    sm1 = op(sm0_)
    assert np.allclose(sm1.states[:, 0], sm0_.states)  # not mixed at all
    assert np.allclose(sm1.states[:, 1], np.mean(sm0_.states, axis=0))  # fully mixed

    # pure T1
    sm0_ = StateMatrix([0, 0, 0])  # start from 0

    op = exchange.X(10, 0, T1=[1e10, 1e-10])
    sm1 = op(sm0_)
    assert np.allclose(sm1.states[0], 0)  # not recovered
    assert np.allclose(sm1.states[1], [0, 0, 1])  # fully recovered

    # mixing T1
    op = exchange.X(10, 1e10, T1=[20, 70])
    sm1 = op(sm0_)
    mean_relax = 1 - np.exp(-np.mean(op.tau / op.T1))
    assert np.allclose(sm1.states[..., 2], mean_relax, atol=1e-5)

    # ndim khi
    khi = exchange.exchange_matrix([1, 1])
    op = exchange.X(1, khi)
    assert op.shape == (2, 2)
    sm1 = op(sm0)
    assert sm1.shape == (2, 2)
