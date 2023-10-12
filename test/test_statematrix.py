import pytest
import itertools
import numpy as np
from epgpy import statematrix, common, transition, shift


def test_state_matrix_class():
    sm = statematrix.StateMatrix(init=[0, 0, 1])
    assert sm.ndim == 1
    assert sm.shape == (1,)
    assert sm.nstate == 0
    assert sm.coords is None
    assert np.allclose(sm.states, [[[0, 0, 1]]])
    assert np.allclose(sm.density, [1])

    sm = statematrix.StateMatrix(init=[[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert sm.ndim == 1
    assert sm.shape == (1,)
    assert sm.nstate == 1
    assert np.allclose(sm.states, [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]])
    assert np.allclose(sm.density, 1)  # density is still 1

    sm = statematrix.StateMatrix(init=[[[0, 0, 1]]] * 5)
    assert sm.ndim == 1
    assert sm.shape == (5,)
    assert sm.nstate == 0
    assert np.allclose(sm.states, [[[0, 0, 1]]] * 5)

    sm = statematrix.StateMatrix(init=[[[[0, 0, 1]]], [[[0, 0, 2]]]])
    assert sm.ndim == 2
    assert sm.shape == (2, 1)
    assert sm.nstate == 0

    # equilibrium
    sm = statematrix.StateMatrix(equilibrium=[[[0, 0, 1]], [[0, 0, 2]]])
    assert sm.ndim == 1
    assert sm.shape == (2,)
    assert sm.nstate == 0
    assert np.allclose(sm.density, [1, 2])
    assert np.allclose(sm.states, sm.equilibrium)

    # density
    sm = statematrix.StateMatrix(density=[1, 3])
    assert sm.ndim == 1
    assert sm.shape == (2,)
    assert sm.nstate == 0
    assert np.allclose(sm.density, [1, 3])
    assert np.allclose(sm.states, sm.equilibrium)

    # sanity checks
    with pytest.raises(ValueError):
        statematrix.StateMatrix(init=[0, 1])
    with pytest.raises(ValueError):
        statematrix.StateMatrix(init=[[0, 0, 0, 1]])
    with pytest.raises(ValueError):
        statematrix.StateMatrix(init=[[[0, 1]]])
    with pytest.raises(ValueError):
        statematrix.StateMatrix(init=[[0, 0, 1], [0, 0, 1]])
    with pytest.raises(ValueError):
        statematrix.StateMatrix(init=[[[0, 0, 1], [0, 0, 1]]])

    # test expand
    sm = statematrix.StateMatrix(density=[1, 3])
    assert sm.shape == (2,)
    sm.expand(3)
    assert sm.shape == (2, 1, 1)
    assert sm.states.shape == (2, 1, 1, 1, 3)
    sm.reduce(1)
    assert sm.shape == (2,)
    # test copy
    sm2 = sm.copy()
    assert sm2.states is not sm.states
    # copy also equilibrium
    assert np.allclose(sm.density, [1, 3])

    # nstate and resize
    sm = statematrix.StateMatrix(nstate=3)
    assert sm.nstate == 3
    sm.resize(5)
    assert sm.nstate == 5
    sm.resize(0)
    assert sm.nstate == 0
    assert np.all(sm == [[[0, 0, 1]]])

    # reshape
    sm = statematrix.StateMatrix(shape=(3,))
    assert sm.shape == (3,)

    # short cuts
    sm = statematrix.StateMatrix(
        init=[[1 / 2, 1 / 2, 0], [1, 1, 0.5], [1 / 2, 1 / 2, 0]]
    )
    assert np.allclose(sm.F, [1 / 2, 1, 1 / 2])
    assert np.allclose(sm.F0, [1])
    assert np.allclose(sm.Z, [0, 0.5, 0])
    assert np.allclose(sm.Z0, [0.5])

    # operators eq, add and iadd
    sm1 = statematrix.StateMatrix([0, 0, 1])
    sm2 = statematrix.StateMatrix([1, 1, 0])
    assert np.all(sm1 == [0, 0, 1])
    assert np.all((sm1 + sm2) == [1, 1, 1])
    sm1 += sm2
    assert np.all(sm1 == [1, 1, 1])

    # as array
    assert np.all(common.asnumpy(sm1) == common.asnumpy(sm1.states))
    assert np.all(sm1 + [1, 1, 1] == [2, 2, 2])

    # options
    sm = statematrix.StateMatrix(max_nstate=3, kgrid=2)
    assert sm.options == {"max_nstate": 3, "kgrid": 2}


def test_conservation():
    """test energy conservation"""
    sm = statematrix.StateMatrix()
    assert np.isclose(sm.norm, 1)

    ops = [transition.T(30, 30), shift.S(1)] * 10
    for op in ops:
        sm = op(sm)
    assert np.isclose(sm.norm, 1)

    sm = statematrix.StateMatrix(equilibrium=[0, 0, 10])
    assert np.isclose(sm.norm, 10)

    for op in ops:
        sm = op(sm)
    assert np.isclose(sm.norm, 10)


# def test_wavenumber():
# init wavenumbers
# sm = statematrix.StateMatrix([0, 0, 1], wavenumbers=[0, 0])
# assert sm.wavenumbers.shape == (1, 1, 2)
#
# sm = statematrix.StateMatrix(nstate=3, wavenumbers=[[0, 0, 0]] * 7)
# assert sm.wavenumbers.shape == (1, 7, 3)
#
# sm = statematrix.StateMatrix(shape=[2, 3], wavenumbers=[[[0]]] * 2)
# assert sm.wavenumbers.shape == (2, 1, 1)
#
# with pytest.raises(ValueError):
#     # invalid kdim
#     statematrix.StateMatrix(wavenumbers=[0, 0, 0, 0])
#
# with pytest.raises(ValueError):
#     # incompatible nstates
#     statematrix.StateMatrix(nstate=3, wavenumbers=[0, 0, 0])
#
# with pytest.raises(ValueError):
#     # incompatible shapes
#     statematrix.StateMatrix(shape=[2], wavenumbers=[[[0]]] * 3)
#
# with pytest.raises(ValueError):
#     # incompatible shapes
#     statematrix.StateMatrix(shape=[2, 3], wavenumbers=[[[0]]] * 3)
#
# # resize
# sm = statematrix.StateMatrix(nstate=1, wavenumbers=[[0, 0, 0]] * 3)
# sm.resize(nstate=2)
# assert sm.wavenumbers.shape == (1, 5, 3)
#
# # broadcast
# sm = statematrix.StateMatrix(wavenumbers=[[[0, 0]]] * 2)
# assert sm.broadcast((2, 3))
# with pytest.raises(ValueError):
#     # incompatible shapes
#     sm.broadcast((3, 2))

# update
# sm = statematrix.StateMatrix(np.ones((1, 3)), wavenumbers=np.ones((1, 3)))
# assert sm.states.shape == (1, 1, 3)
# assert sm.wavenumbers.shape[:1] == (1,)
#
# # update states shape
# sm.states = np.ones((2, 1, 3))
# sm.update()
# assert sm.wavenumbers.shape[:1] == (2,) # broadcast
#
# # update states shape
# sm.wavenumbers = np.ones((2, 3, 1, 3))
# sm.update()
# assert sm.states.shape[:2] == (2, 3) # broadcast


# def test_format_wavenumbers():
#
#     states = statematrix.format_states([0, 0, 1])
#     wnum = statematrix.format_wavenumbers(states, [0])
#     assert wnum.shape == (1, 1, 1)
#     assert np.allclose(wnum, 0)
#
#     states = statematrix.format_states([0, 0, 1])
#     wnum = statematrix.format_wavenumbers(states, [0, 0])
#     assert wnum.shape == (1, 1, 2)
#     assert np.allclose(wnum, 0)
#
#     states = statematrix.format_states([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
#     wnum = statematrix.format_wavenumbers(states, [[1, 1], [0, 0], [-1, -1]])
#     assert wnum.shape == (1, 3, 2)
#
#     states = statematrix.format_states([[[0, 0, 1]]] * 4)
#     wnum = statematrix.format_wavenumbers(states, [1, 1, 1])
#     assert wnum.shape == (1, 1, 3)
#
#     states = statematrix.format_states([0, 0, 1])
#     with pytest.raises(ValueError):
#         statematrix.format_wavenumbers(states, [[-1], [0], [1]])
#
#     states = statematrix.format_states([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
#     with pytest.raises(ValueError):
#         statematrix.format_wavenumbers(states, [0, 0, 0])
#
#     states = statematrix.format_states([[[0, 0, 1]]] * 4)
#     with pytest.raises(ValueError):
#         statematrix.format_wavenumbers(states, [[[1]]] * 3)
