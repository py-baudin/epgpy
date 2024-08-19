import numpy as np
import pytest
from epgpy import opscalar, opmatrix, statematrix, transition, evolution

StateMatrix = statematrix.StateMatrix


def test_MatrixOp_class():
    ScalarOp = opscalar.ScalarOp
    MatrixOp = opmatrix.MatrixOp
    rstate = np.random.RandomState(0)

    mat = [[0, 1j, 0], [-1j, 0, 0], [0, 0, 1]]
    op = MatrixOp(mat)

    assert op.nshift == 0
    assert op.shape == (1,)

    sm0 = StateMatrix([1, 1, 1])
    assert np.allclose(op(sm0).states, [1j, -1j, 1])

    # multiple dimensions
    op = MatrixOp([mat, mat])
    assert op.shape == (2,)
    assert np.allclose(op(sm0).states, [[[1j, -1j, 1]], [[1j, -1j, 1]]])

    # with constant
    const = np.diag([0, 0, 0.5])
    op = MatrixOp([mat, mat], [const, const])
    assert op.shape == (2,)
    assert np.allclose(op(sm0).states, [[[1j, -1j, 1.5]], [[1j, -1j, 1.5]]])

    # using axis
    op = MatrixOp([mat, mat], axes=2)
    assert op.shape == (1, 1, 2)

    op = MatrixOp(3 * [2 * [mat]], axes=(1, 3))
    assert op.shape == (1, 3, 1, 2)

    # abritrary matrices
    mat = rstate.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat += mat[..., (1, 0, 2), :][..., (1, 0, 2)].conj()
    op = MatrixOp(mat)
    assert np.allclose(op(sm0).states, mat @ [1, 1, 1])

    mat0 = rstate.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat0 += mat0[..., (1, 0, 2), :][..., (1, 0, 2)].conj()
    op = MatrixOp(mat, mat0)
    assert np.allclose(op(sm0).states, mat @ [1, 1, 1] + mat0 @ [0, 0, 1])

    # combine
    mat_ = rstate.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat_ += mat_[..., (1, 0, 2), :][..., (1, 0, 2)].conj()
    mat0_ = rstate.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat0_ += mat0_[..., (1, 0, 2), :][..., (1, 0, 2)].conj()

    op = MatrixOp(mat) @ MatrixOp(mat_)
    assert np.allclose(op.mat, mat_ @ mat)
    assert op.mat0 is None

    op = MatrixOp(mat, mat0) @ MatrixOp(mat_)
    assert np.allclose(op.mat, mat_ @ mat)
    assert np.allclose(op.mat0, mat_ @ mat0)

    op = MatrixOp(mat) @ MatrixOp(mat_, mat0_)
    assert np.allclose(op.mat, mat_ @ mat)
    assert np.allclose(op.mat0, mat0_)

    op = MatrixOp(mat, mat0) @ MatrixOp(mat_, mat0_)
    assert np.allclose(op.mat, mat_ @ mat)
    assert np.allclose(op.mat0, mat_ @ mat0 + mat0_)

    op = MatrixOp(mat_, mat0_) @ MatrixOp(mat, mat0)
    assert np.allclose(op.mat, mat @ mat_)
    assert np.allclose(op.mat0, mat @ mat0_ + mat0)

    # combine with scalar op
    coeff = rstate.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff += coeff[..., (1, 0, 2)].conj()
    coeff0 = rstate.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff0 += coeff0[..., (1, 0, 2)].conj()

    op = MatrixOp(mat, mat0) @ ScalarOp(coeff, coeff0)
    assert np.allclose(op.mat, np.diag(coeff) @ mat)
    assert np.allclose(op.mat0, np.diag(coeff) @ mat0 + np.diag(coeff0))

    op = ScalarOp(coeff, coeff0) @ MatrixOp(mat, mat0)
    assert np.allclose(op.mat, mat @ np.diag(coeff))
    assert np.allclose(op.mat0, mat @ np.diag(coeff0) + mat0)

    # combine and broadcasting
    mat1 = rstate.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat1 += mat1[..., [1, 0, 2], :][..., [1, 0, 2]].conj()
    mat2 = rstate.uniform(-1, 1, (2, 3, 3, 2)).dot([1, 1j])
    mat2 += mat2[..., [1, 0, 2], :][..., [1, 0, 2]].conj()
    op12 = MatrixOp(mat1) @ MatrixOp(mat2)
    assert op12.shape == (2,)

    arr3 = rstate.uniform(-1, 1, (1, 4, 3, 2)).dot([1, 1j])
    arr3 += arr3[..., [1, 0, 2]].conj()
    op13 = MatrixOp(mat2) @ ScalarOp(arr3)
    assert op13.shape == (2, 4)

    mat3 = rstate.uniform(-1, 1, (3, 3, 3, 2)).dot([1, 1j])
    mat3 += mat3[..., [1, 0, 2], :][..., [1, 0, 2]].conj()
    with pytest.raises(ValueError):
        MatrixOp(mat2) @ MatrixOp(mat3)


def test_MatrixOp_diff():
    ScalarOp = opscalar.ScalarOp
    MatrixOp = opmatrix.MatrixOp

    rstate = np.random.RandomState(0)
    A = rstate.uniform(-1, 1, (3, 3))
    A += A[[1, 0, 2], :][:, [1, 0, 2]].conj()
    B = rstate.uniform(-1, 1, (3, 3))
    B += B[[1, 0, 2], :][:, [1, 0, 2]].conj()
    dA = rstate.uniform(-1, 1, (3, 3))
    dA += dA[[1, 0, 2], :][:, [1, 0, 2]].conj()
    dB = rstate.uniform(-1, 1, (3, 3))
    dB += dB[[1, 0, 2], :][:, [1, 0, 2]].conj()

    a = rstate.uniform(-1, 1, 3)
    a += a[[1, 0, 2]].conj()
    b = rstate.uniform(-1, 1, 3)
    b += b[[1, 0, 2]].conj()
    da = rstate.uniform(-1, 1, 3)
    da += da[[1, 0, 2]].conj()
    db = rstate.uniform(-1, 1, 3)
    db += db[[1, 0, 2]].conj()

    op1 = MatrixOp(A, B, dmats={"x": (dA, dB)}, order1="x", order2=[("x", "y")])
    op2 = ScalarOp(a, b, darrs={"y": (da, db)}, order1="y", order2=[("x", "y")])

    sm0 = StateMatrix([1, 1, 0])
    sm2 = op1(op2(op1(sm0)))

    # combine
    op12 = op1 @ op2 @ op1
    assert op12.parameters_order1 == {"x", "y"}
    sm2_ = op12(sm0)
    assert np.allclose(sm2.states, sm2_.states)
    assert np.allclose(sm2.order1["x"].states, sm2_.order1["x"].states)
    assert np.allclose(sm2.order1["y"].states, sm2_.order1["y"].states)
    assert np.allclose(sm2.order2[("x", "y")].states, sm2_.order2[("x", "y")].states)

    # finite diffs
    dx = 1e-5j
    A_x = A + dx * dA
    B_x = B + dx * dB
    op1_x = MatrixOp(A_x, B_x, check=False)
    op12_x = op1_x.combine(op2, check=False).combine(op1_x, check=False)
    sm2_x = op12_x(sm0)
    fdiff_x = (sm2_x.states - sm2.states).imag * 1e5
    assert np.allclose(fdiff_x, sm2.order1["x"].states)

    fdiff2_xy = (sm2_x.order1["y"].states - sm2.order1["y"].states).imag * 1e5
    assert np.allclose(fdiff2_xy, sm2.order2[("x", "y")].states)

    # copy
    op2 = op1.copy(name="op2")
    assert np.allclose(op2.mat, op1.mat)
    assert np.allclose(op2.mat0, op1.mat0)
    assert [
        np.allclose(op2.dmats[param][0], op1.dmats[param][0]) for param in op1.dmats
    ]
