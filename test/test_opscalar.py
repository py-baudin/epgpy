import numpy as np
import pytest
from epgpy import opscalar, statematrix

StateMatrix = statematrix.StateMatrix


def test_ScalarOp_class():
    ScalarOp = opscalar.ScalarOp

    arr = [1j, -1j, 0.5]
    op = ScalarOp(arr)

    assert op.nshift == 0
    assert op.shape == (1,)

    sm0 = StateMatrix([1, 1, 1])
    assert np.allclose(op(sm0).states, [1j, -1j, 0.5])

    # multiple dimensions
    op = ScalarOp([arr, arr])
    assert op.shape == (2,)
    assert np.allclose(op(sm0).states, [[[1j, -1j, 0.5]], [[1j, -1j, 0.5]]])

    # with constant
    const = [0, 0, 0.5]
    op = ScalarOp([arr, arr], [const, const])
    assert op.shape == (2,)
    assert np.allclose(op(sm0).states, [[[1j, -1j, 1]], [[1j, -1j, 1]]])

    # using axis
    op = ScalarOp([arr, arr], axes=2)
    assert op.shape == (1, 1, 2)

    op = ScalarOp(3 * [2 * [arr]], axes=(1, 3))
    assert op.shape == (1, 3, 1, 2)

    # abritrary matrices
    arr = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    arr += arr[..., (1, 0, 2)].conj()
    op = ScalarOp(arr)
    assert np.allclose(op(sm0).states, arr * [1, 1, 1])

    arr0 = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    arr0 += arr0[..., (1, 0, 2)].conj()
    op = ScalarOp(arr, arr0)
    assert np.allclose(op(sm0).states, arr * [1, 1, 1] + arr0 * [0, 0, 1])

    # mat property
    assert np.allclose(op.mat[0], np.diag(arr))
    assert np.allclose(op.mat0[0], np.diag(arr0))

    # combine
    arr_ = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    arr_ += arr_[..., (1, 0, 2)].conj()
    arr0_ = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    arr0_ += arr0_[..., (1, 0, 2)].conj()

    op = ScalarOp(arr) @ ScalarOp(arr_)
    assert np.allclose(op.arr, arr_ * arr)
    assert op.arr0 is None

    op = ScalarOp(arr, arr0, name="a") @ ScalarOp(arr_, name="b")
    assert np.allclose(op.arr, arr_ * arr)
    assert np.allclose(op.arr0, arr_ * arr0)

    op = ScalarOp(arr) @ ScalarOp(arr_, arr0_)
    assert np.allclose(op.arr, arr_ * arr)
    assert np.allclose(op.arr0, arr0_)

    op = ScalarOp(arr, arr0) @ ScalarOp(arr_, arr0_)
    assert np.allclose(op.arr, arr_ * arr)
    assert np.allclose(op.arr0, arr_ * arr0 + arr0_)

    # copy
    op2 = op.copy(name="op2")
    assert np.allclose(op2.arr, op.arr)
    assert np.allclose(op2.arr0, op.arr0)


def test_ScalarOp_diff1():
    ScalarOp = opscalar.ScalarOp

    r1, r2 = np.exp(-0.5), np.exp(-0.1)
    arr = [r2, r2, r1]
    arr0 = [0, 0, 1 - r1]
    darrs = {
        "r2": ([-r2, -r2, 0], None),
        "r1": ([0, 0, -r1], [0, 0, r1]),
    }

    # order1
    op = ScalarOp(arr, arr0, darrs=darrs, order1=["r1", "r2"], name="op1")

    sm0 = StateMatrix([1, 1, 0])
    sm1 = op(sm0)

    # finite differences

    # r2
    arr_r2 = [r2 * np.exp(-1e-5 * 1j), r2 * np.exp(1e-5 * 1j), r1]
    op_r2 = ScalarOp(arr_r2, arr0, name="op_r2")
    diff_r2 = (op_r2(sm0).states - sm1.states).imag * 1e5
    assert np.allclose(diff_r2[..., (0, 2)], sm1.order1["r2"].states[..., (0, 2)])

    # r1
    arr_r1 = [r2, r2, r1 * np.exp(-1j * 1e-5)]
    arr0_r1 = [0, 0, 1 - r1 * np.exp(-1j * 1e-5)]
    op_r1 = ScalarOp(arr_r1, arr0_r1, name="op_r1", check=False)
    diff_r1 = (op_r1(sm0).states - sm1.states).imag * 1e5
    assert np.allclose(diff_r1, sm1.order1["r1"].states)

    # combine
    op2 = op @ op
    sm2 = op2(sm0)
    assert np.allclose(sm2.states, op(op(sm0)).states)
    assert np.allclose(sm2.order1["r2"].states, op(op(sm0)).order1["r2"].states)
    assert np.allclose(sm2.order1["r2"], [-2 * r2**2, -2 * r2**2, 0])
    assert np.allclose(sm2.order1["r1"].states, op(op(sm0)).order1["r1"].states)
    assert np.allclose(sm2.order1["r1"], [0, 0, 2 * r1**2])

    sm3 = (op @ op2)(sm0)
    sm3_ = op(op(op(sm0)))
    assert np.allclose(sm3.states, sm3_.states)
    assert np.allclose(sm3.order1["r1"].states, sm3_.order1["r1"].states)
    assert np.allclose(sm3.order1["r2"].states, sm3_.order1["r2"].states)

    # finite differences

    # r2
    op3_r2 = op_r2.combine(op_r2).combine(op_r2)
    diff_r2 = (op3_r2(sm0).states - sm3.states).imag * 1e5
    assert np.allclose(diff_r2[..., (0, 2)], sm3.order1["r2"].states[..., (0, 2)])

    # r1
    op3_r1 = op_r1.combine(op_r1, check=False).combine(op_r1, check=False)
    diff_r1 = (op3_r1(sm0).states - sm1.states).imag * 1e5
    assert np.allclose(diff_r1, sm3.order1["r1"].states)


def test_ScalarOp_diff2():
    ScalarOp = opscalar.ScalarOp

    r1, r2 = np.exp(-0.1), np.exp(-0.5)
    arr = [r2, r2, r1]
    arr0 = [0, 0, 1 - r1]
    darrs = {
        "r2": ([-r2, -r2, 0], None),
        "r1": ([0, 0, -r1], [0, 0, r1]),
        "t": ([-0.5 * r2, -0.5 * r2, -0.1 * r1], [0, 0, 0.1 * r1]),
    }
    d2arrs = {
        ("t", "r2"): ([(0.5 - 1) * r2, (0.5 - 1) * r2, 0], None),
        ("t", "r1"): ([0, 0, (0.1 - 1) * r1], [0, 0, -(0.1 - 1) * r1]),
    }

    # order2
    op = ScalarOp(
        arr,
        arr0,
        darrs=darrs,
        d2arrs=d2arrs,
        order1=["t", "r1", "r2"],
        order2={("t", "r2"), ("t", "r1")},
        name="op1",
    )

    sm0 = StateMatrix([1, 1, 0])
    sm1 = op(sm0)

    # finite differences
    dt = 1j * 1e-5
    dr1 = np.exp(-0.1 * dt)
    dr2 = np.exp(-0.5 * dt)
    arr_t = [r2 * dr2, r2 * dr2, r1 * dr1]
    arr0_t = [0, 0, 1 - r1 * dr1]
    op_t = ScalarOp(
        arr_t,
        arr0_t,
        name="op_t",
        order1=["r1", "r2"],
        darrs={
            "r2": ([-(1 + dt) * r2 * dr2, -(1 + dt) * r2 * dr2, 0], None),
            "r1": ([0, 0, -(1 + dt) * r1 * dr1], [0, 0, (1 + dt) * r1 * dr1]),
        },
        check=False,
    )

    # order 1
    diff_t = (op_t(sm0).states - sm1.states).imag * 1e5
    assert np.allclose(diff_t[..., (0, 2)], sm1.order1["t"].states[..., (0, 2)])

    # order 2
    diff_t_r2 = (
        op_t.derive1(sm0, "r2").states - op.derive1(sm0, "r2").states
    ).imag * 1e5
    assert np.allclose(sm1.order2[("r2", "t")].states, diff_t_r2)

    diff_t_r1 = (
        op_t.derive1(sm0, "r1").states - op.derive1(sm0, "r1").states
    ).imag * 1e5
    assert np.allclose(sm1.order2[("r1", "t")].states, diff_t_r1)

    # combine
    op3 = op @ op @ op
    sm3 = op3(sm0)
    sm3_ = op(op(op(sm0)))
    assert np.allclose(sm3.states, sm3_.states)
    assert np.allclose(sm3.order2[("r2", "t")], sm3_.order2[("r2", "t")])
    assert np.allclose(sm3.order2[("r1", "t")], sm3_.order2[("r1", "t")])

    # finite diff
    op3_t = op_t.combine(op_t, check=False).combine(op_t, check=False)
    diff3_t = (op3_t(sm0).states - sm3.states).imag * 1e5
    assert np.allclose(diff3_t[..., (0, 2)], sm3.order1["t"].states[..., (0, 2)])

    diff3_t_r2 = (
        op3_t.derive1(sm0, "r2").states - op3.derive1(sm0, "r2").states
    ).imag * 1e5
    assert np.allclose(sm3.order2[("r2", "t")].states, diff3_t_r2)

    diff3_t_r1 = (
        op3_t.derive1(sm0, "r1").states - op3.derive1(sm0, "r1").states
    ).imag * 1e5
    assert np.allclose(sm3.order2[("r1", "t")].states, diff3_t_r1)

    # copy
    op2 = op.copy(name="op2")
    assert np.allclose(op2.arr, op.arr)
    assert np.allclose(op2.arr0, op.arr0)
    assert [np.allclose(op2.darrs[param][0], op.darrs[param][0]) for param in op.darrs]
    assert [np.allclose(op2.d2arrs[pair][0], op.d2arrs[pair][0]) for pair in op.d2arrs]


def test_scalarop_ndim():
    ScalarOp = opscalar.ScalarOp

    def stack(ar1, ar2, ar3):
        return np.stack(np.broadcast_arrays(ar1, ar2, ar3), axis=-1)

    r1 = np.asarray([np.exp(-0.5)] * 2)
    r2 = np.asarray([[np.exp(-0.1)]] * 3)
    arr = stack(r2, r2, r1)
    arr0 = stack(0, 0, 1 - r1)
    darrs = {
        "r2": (stack(-r2, -r2, 0), None),
        "r1": (stack(0, 0, -r1), stack(0, 0, r1)),
    }
    d2arrs = {
        ("r2", "r2"): (stack(r2, r2, 0), None),
        ("r1", "r1"): (stack(0, 0, r1), stack(0, 0, -r1)),
    }

    # order1
    op = ScalarOp(arr, arr0, darrs=darrs, d2arrs=d2arrs, order2=True, name="op1")
    sm0 = statematrix.StateMatrix([1, 1, 0.5])
    sm1 = op(sm0)

    assert sm1.shape == op.shape == (3, 2)
    assert sm1.order1["r1"].shape == sm1.order1["r2"].shape == (3, 2)
    assert sm1.order2[("r1", "r1")].shape == sm1.order2[("r2", "r2")].shape == (3, 2)

    assert np.allclose(sm1.states, sm1.states[0, 0])
    assert np.allclose(sm1.order1["r1"].states, sm1.order1["r1"].states[0, 0])
    assert np.allclose(sm1.order1["r2"].states, sm1.order1["r2"].states[0, 0])
    assert np.allclose(
        sm1.order2[("r1", "r1")].states, sm1.order2[("r1", "r1")].states[0, 0]
    )
    assert np.allclose(
        sm1.order2[("r2", "r2")].states, sm1.order2[("r2", "r2")].states[0, 0]
    )
