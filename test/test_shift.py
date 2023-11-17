import pytest
import numpy as np
from epgpy import shift, statematrix
from epgpy import operators, transition


def test_shift():
    sm0 = np.asarray([[1, 1, 0]])

    sm1 = shift.shift1d(sm0, 1, inplace=False)
    assert np.allclose(sm1, [[0, 1, 0], [0, 0, 0], [1, 0, 0]])

    shifts = [(-1) ** i * i for i in range(1, 10)]
    shifts += [-i for i in shifts]
    sm1 = sm0
    for k in shifts:
        sm1 = shift.shift1d(sm1, k, inplace=False)
    nstate = (len(sm1) - 1) // 2
    assert np.allclose(sm1[nstate], [1, 1, 0])
    assert np.allclose(sm1[:nstate], 0)
    assert np.allclose(sm1[nstate + 1 :], 0)

    nmax = shift.get_nmax(shifts)
    sm1 = sm0
    for k in shifts:
        sm1 = shift.shift1d(sm1, k, inplace=False, nmax=nmax)
    nstate2 = (len(sm1) - 1) // 2
    assert nmax == nstate2 < nstate
    assert np.allclose(sm1[nstate2], [1, 1, 0])
    assert np.allclose(sm1[:nstate2], 0)
    assert np.allclose(sm1[nstate2 + 1 :], 0)


def test_shiftnd():
    # initial conditions
    sm0 = np.asarray([[1, 1, 0]])
    k0 = np.array([[0, 0, 0]])

    # 1d case
    sm1_ = shift.shift1d(sm0, 1, inplace=False)
    sm1, k1 = shift.shiftnd(sm0, k0, [1, 0, 0])
    assert np.allclose(sm1_, sm1)
    assert np.allclose(k1, [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])

    sm2, k2 = shift.shiftnd(sm1, k1, [-1, 0, 0])
    assert np.allclose(sm2, [[1, 1, 0]])
    assert np.allclose(k2, 0)

    # 3d sequence
    shifts = [[i, j, k] for i in [-1, 2] for j in [-2, 1] for k in [2, -2]]
    shifts += [[-i, -j, -k] for i, j, k in shifts]
    sm1, k1 = sm0, k0
    for dk in shifts:
        sm1, k1 = shift.shiftnd(sm1, k1, dk)
    assert np.allclose(sm1, [[1, 1, 0]])
    assert np.allclose(k2, 0)

    # multi-dim shift
    sm0 = np.asarray([[[1, 1, 0]], [[1j, -1j, 0]]])
    k0 = np.array([[[0, 0]], [[0, 0]]])

    sm1, k1 = shift.shiftnd(sm0, k0, [[1, -1], [2, 0]])
    # warning: due to sorting of k
    # multi-dim and single dim may return states in different orders
    sm11, k11 = shift.shiftnd(sm0[0], k0[0], [[1, -1]])
    sm12, k12 = shift.shiftnd(sm0[1], k0[1], [[2, 0]])
    assert np.allclose(sm11, sm1[0])
    assert np.allclose(sm12, sm1[1])
    assert np.allclose(k11, k1[0])
    assert np.allclose(k12, k1[1])


def test_shiftmerge():
    # initial conditions
    sm0 = np.asarray([[1, 1, 0]])
    k0 = np.array([[0, 0, 0]])

    # ref
    sm1_, k1_ = shift.shiftnd(sm0, k0, [1, 0, 0])
    sm2_, k2_ = shift.shiftnd(sm1_, k1_, [-1, 0, 0])

    # integer non-merging phase-states
    sm1, k1 = shift.shiftmerge(sm0, k0, [1, 0, 0])
    assert np.allclose(sm1, sm1_)
    assert np.allclose(k1, k1_)

    sm2, k2 = shift.shiftnd(sm1, k1, [-1, 0, 0])
    assert np.allclose(sm2, sm2_)
    assert np.allclose(k2, k2_)

    # non-integer phase-states
    sm1, k1 = shift.shiftmerge(sm0, k0, [1.1, 0, 0])
    assert np.allclose(sm1, sm1_)
    assert np.allclose(k1, [[-1.1, 0, 0], [0, 0, 0], [1.1, 0, 0]])
    sm2, k2 = shift.shiftmerge(sm1, k1, [-1.1, 0, 0])
    assert np.allclose(sm2, sm2_)
    assert np.allclose(k2, 0)  # only zero-th state

    # merging phase states
    sm0 = np.asarray([[0, 0, 1], [1, 1, 0], [0, 0, 1]])
    k0 = np.asarray([[-0.9, 0, 0], [0, 0, 0], [0.9, 0, 0]])
    sm1, k1 = shift.shiftmerge(sm0, k0, [1.1, 0, 0])
    assert np.allclose(sm1, [[0, 1, 1], [0, 0, 0], [1, 0, 1]])
    assert np.allclose(k1, [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])

    sm0 = np.asarray([[0.5, 0, 0], [1, 1, 0], [0, 0.5, 0]])
    k0 = np.asarray([[-2.1, 0, 0], [0, 0, 0], [2.1, 0, 0]])
    sm1, k1 = shift.shiftmerge(sm0, k0, [0.9, 0, 0])
    assert np.allclose(sm1, [[0.5, 1, 0], [0, 0, 0], [1, 0.5, 0]])
    assert np.allclose(k1, [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])

    # inverse gradient
    sm2, k2 = shift.shiftmerge(sm1, k1, [-1, 0, 0])
    assert np.allclose(sm0, sm2)
    # note: due to the merging approximation, k2 != k0

    # hyper echo
    necho = 100
    rot1 = transition.rotation_operator(30, 45)[0]
    rot2 = transition.rotation_operator(-30, -45)[0]
    ref = transition.rotation_operator(180, 0)[0]
    sh = np.array([1.9, -1.8, 0])
    sm = np.array([[1, 1, 0]])
    k = np.array([[0, 0, 0]])
    for _ in range(necho):
        sm = sm @ rot1.T
        sm, k = shift.shiftmerge(sm, k, sh)
    sm = sm @ ref.T
    for _ in range(necho):
        sm, k = shift.shiftmerge(sm, k, sh)
        sm = sm @ rot2.T
    assert np.allclose(sm, [[1, 1, 0]])
    assert np.allclose(k, [[0, 0, 0]])


def test_S_class():
    S, C = shift.S, shift.C
    SM = statematrix.StateMatrix

    # int 1d shift
    shift1d = S(-2)
    assert shift1d.nshift == 2
    assert shift1d.k == -2
    assert shift1d.shape == (1,)

    sm0 = SM([1, 1, 0], max_nstate=1)
    sm1 = S(1)(sm0)
    assert np.allclose(sm1.states, [[[0, 1, 0], [0, 0, 0], [1, 0, 0]]])

    sm2 = S(-1)(sm1)
    assert np.allclose(sm2.states, [[[0, 0, 0], [1, 1, 0], [0, 0, 0]]])

    # int-nd
    sm0 = SM([1, 1, 0])
    shiftnd = S([1, 0, 0])
    assert np.allclose(shiftnd.k, [[1, 0, 0]])
    assert shiftnd.shape == (1,)
    sm1 = shiftnd(sm0)
    assert sm1.kdim == 3
    assert sm1.nstate == 1
    assert np.allclose(sm1.states, [[0, 1, 0], [0, 0, 0], [1, 0, 0]])

    # multi-dim
    shiftnd = S([[1, 0, 0], [2, 0, 0]])
    assert shiftnd.shape == (2,)
    sm1 = shiftnd(sm0)
    assert sm1.shape == (2,)  # shape broadcasting
    assert sm1.kdim == 3
    assert sm1.nstate == 1
    assert np.allclose(sm1.states, [[[0, 1, 0], [0, 0, 0], [1, 0, 0]]])
    assert np.allclose(sm1.k[0] * 2, sm1.k[1])

    # int 1d to nd
    shift1 = S(1)
    shift2 = S([0, -2, 1])
    sm1 = shift2(shift1(sm0))
    assert np.allclose(sm1.k, [[-1, 2, -1], [0, 0, 0], [1, -2, 1]])
    assert np.allclose(shift1(sm1).k, [[-2, 2, -1], [0, 0, 0], [2, -2, 1]])

    # float nd
    shiftndf = S([1.2, 0, -0.3])
    assert shiftndf.shape == (1,)
    assert shiftndf.kdim == 3
    with pytest.raises(AttributeError):
        shiftndf(sm0)  # grid size not set

    sm0 = SM([1, 1, 0], kgrid=0.1)  # set grid size to 0.1 (eg. rad/m)
    sm1 = shiftndf(sm0)
    assert sm1.kdim == 3
    assert sm1.nstate == 1
    assert np.allclose(sm1.k, [[-1.2, 0, 0.3], [0, 0, 0], [1.2, 0, -0.3]])

    # int 1d to nd to float
    shift3 = S([0.2, 0, -0.3])
    sm1 = shift3(shift1(sm0))  # [1, 0., 0] + [0.2, 0, -0.3]
    assert np.allclose(sm1.k, [[-1.2, 0, 0.3], [0, 0, 0], [1.2, 0, -0.3]])
    sm1 = shift3(shift2(shift1(sm0)))  # [1, 0., 0] + [0, -2, 1] + [0.2, 0, -0.3]
    assert np.allclose(sm1.k, [[-1.2, 2, -0.7], [0, 0, 0], [1.2, -2, 0.7]])
    assert np.allclose(shift1(sm1).k, [[-2.2, 2, -0.7], [0, 0, 0], [2.2, -2, 0.7]])

    #
    # time accumulation
    shift4 = C(10)
    sm0 = SM([1, 1, 0], kvalue=0.1)
    sm1 = shift4(sm0)
    assert np.allclose(sm1.t, [[-10, 0, 10]])

    sm1 = shift4(shift2(sm0))
    assert np.allclose(sm1.k, [[0, -0.2, 0.1], [0, 0, 0], [0, 0.2, -0.1]])
    assert np.allclose(sm1.t, [[10, 0, -10]])
    sm2 = shift2(shift4(sm0))
    assert np.allclose(sm1.states, sm2.states)
    assert np.allclose(sm1.coords, sm2.coords)


def test_hyperecho():
    """test hyperecho with various shift methods"""
    T = operators.T
    S = shift.S
    SM = statematrix.StateMatrix

    necho = 100
    exc, ref = T(90, 90), T(180, 0)
    alphas = np.linspace(10, 80, necho)

    # 1d shift
    grad = S(1)
    seq = [exc] + sum([[grad, T(a, 0)] for a in alphas], start=[])
    seq += (
        [grad, ref] + sum([[grad, T(-a, 0)] for a in alphas[::-1]], start=[]) + [grad]
    )

    sm = SM()
    for op in seq:
        sm = op(sm)
    assert np.allclose(sm.states[:, sm.nstate], [1, 1, 0])
    assert np.allclose(sm.states[:, : sm.nstate], 0)

    # nd shift
    grad = S([1, -2, 0])
    seq = [exc] + sum([[grad, T(a, 0)] for a in alphas], start=[])
    seq += (
        [grad, ref] + sum([[grad, T(-a, 0)] for a in alphas[::-1]], start=[]) + [grad]
    )

    sm = SM()
    for op in seq:
        sm = op(sm)
    assert np.allclose(sm.states[:, sm.nstate], [1, 1, 0])
    assert np.allclose(sm.states[:, : sm.nstate], 0)
    # nd shift
    grad = S([1, -2, 0])
    seq = [exc] + sum([[grad, T(a, 0)] for a in alphas], start=[])
    seq += (
        [grad, ref] + sum([[grad, T(-a, 0)] for a in alphas[::-1]], start=[]) + [grad]
    )

    sm = SM()
    for op in seq:
        sm = op(sm)
    assert np.allclose(sm.states[:, sm.nstate], [1, 1, 0])
    assert np.allclose(sm.states[:, : sm.nstate], 0)

    # shift-merge
    grad = S([1.11, -2.29, 0.41])
    seq = [exc] + sum([[grad, T(a, 0)] for a in alphas], start=[])
    seq += (
        [grad, ref] + sum([[grad, T(-a, 0)] for a in alphas[::-1]], start=[]) + [grad]
    )

    sm = SM(kgrid=1)
    for op in seq:
        sm = op(sm)
    assert np.allclose(sm.states[:, sm.nstate], [1, 1, 0])
    assert np.allclose(sm.states[:, : sm.nstate], 0)
