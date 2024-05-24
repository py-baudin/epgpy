import numpy as np
import pytest
from epgpy import oplinear, statematrix

StateMatrix = statematrix.StateMatrix


def test_ScalarOp_class():
    ScalarOp = oplinear.ScalarOp

    coeff = [1j, -1j, 0.5]
    op = ScalarOp(coeff)

    assert op.nshift == 0
    assert op.shape == (1,)

    sm0 = StateMatrix([1, 1, 1])
    assert np.allclose(op(sm0).states, [1j, -1j, 0.5])

    # multiple dimensions
    op = ScalarOp([coeff, coeff])
    assert op.shape == (2,)
    assert np.allclose(op(sm0).states, [[[1j, -1j, 0.5]], [[1j, -1j, 0.5]]])

    # with constant
    const = [0, 0, 0.5]
    op = ScalarOp([coeff, coeff], [const, const])
    assert op.shape == (2,)
    assert np.allclose(op(sm0).states, [[[1j, -1j, 1]], [[1j, -1j, 1]]])

    # using axis
    op = ScalarOp([coeff, coeff], axes=2)
    assert op.shape == (1, 1, 2)

    op = ScalarOp(3 * [2 * [coeff]], axes=(1, 3))
    assert op.shape == (1, 3, 1, 2)

    # abritrary matrices
    coeff = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff += coeff[..., (1, 0, 2)].conj()
    op = ScalarOp(coeff)
    assert np.allclose(op(sm0).states, coeff * [1, 1, 1])

    coeff0 = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff0 += coeff0[..., (1, 0, 2)].conj()
    op = ScalarOp(coeff, coeff0)
    assert np.allclose(op(sm0).states, coeff * [1, 1, 1] + coeff0 * [0, 0, 1])

    # mat property
    assert np.allclose(op.mat[0], np.diag(coeff))
    assert np.allclose(op.mat0[0], np.diag(coeff0))

    # combine
    coeff_ = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff_ += coeff_[..., (1, 0, 2)].conj()
    coeff0_ = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff0_ += coeff0_[..., (1, 0, 2)].conj()

    op = ScalarOp(coeff) @ ScalarOp(coeff_)
    assert np.allclose(op.coeff, coeff_ * coeff)
    assert op.coeff0 is None

    op = ScalarOp(coeff, coeff0) @ ScalarOp(coeff_)
    assert np.allclose(op.coeff, coeff_ * coeff)
    assert np.allclose(op.coeff0, coeff_ * coeff0)

    op = ScalarOp(coeff) @ ScalarOp(coeff_, coeff0_)
    assert np.allclose(op.coeff, coeff_ * coeff)
    assert np.allclose(op.coeff0, coeff0_)

    op = ScalarOp(coeff, coeff0) @ ScalarOp(coeff_, coeff0_)
    assert np.allclose(op.coeff, coeff_ * coeff)
    assert np.allclose(op.coeff0, coeff_ * coeff0 + coeff0_)


def test_MatrixOp_class():
    ScalarOp = oplinear.ScalarOp
    MatrixOp = oplinear.MatrixOp

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
    mat = np.random.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat += mat[..., (1, 0, 2), :][..., (1, 0, 2)].conj()
    op = MatrixOp(mat)
    assert np.allclose(op(sm0).states, mat @ [1, 1, 1])

    mat0 = np.random.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat0 += mat0[..., (1, 0, 2), :][..., (1, 0, 2)].conj()
    op = MatrixOp(mat, mat0)
    assert np.allclose(op(sm0).states, mat @ [1, 1, 1] + mat0 @ [0, 0, 1])

    # combine
    mat_ = np.random.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
    mat_ += mat_[..., (1, 0, 2), :][..., (1, 0, 2)].conj()
    mat0_ = np.random.uniform(-1, 1, (3, 3, 2)).dot([1, 1j])
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
    coeff = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff += coeff[..., (1, 0, 2)].conj()
    coeff0 = np.random.uniform(-1, 1, (3, 2)).dot([1, 1j])
    coeff0 += coeff0[..., (1, 0, 2)].conj()

    op = MatrixOp(mat, mat0) @ ScalarOp(coeff, coeff0)
    assert np.allclose(op.mat, np.diag(coeff) @ mat)
    assert np.allclose(op.mat0, np.diag(coeff) @ mat0 + np.diag(coeff0))

    op = ScalarOp(coeff, coeff0) @ MatrixOp(mat, mat0)
    assert np.allclose(op.mat, mat @ np.diag(coeff))
    assert np.allclose(op.mat0, mat @ np.diag(coeff0) + mat0)


""" TODO: combine

# combine
op1 = bloch.B([[mat] * 2], [[const] * 2])
assert op1.shape == (1, 2)
# const2 = [0,0,0.2]
const2 = np.diag([0, 0, 0.2])
op2 = bloch.B([mat] * 3, [const2] * 3, duration=2)
assert op2.shape == (3,)

op12 = bloch.B.combine([op1, op2])
assert op12.shape == (3, 2)
assert op12.duration == 2
assert np.allclose(op12(sm0).states, op2(op1(sm0)).states)
# breakpoint()
assert op12(StateMatrix(shape=(3,)))

with pytest.raises(ValueError):
    # invalid sm.shape
    op12(StateMatrix(shape=(2,)))

with pytest.raises(ValueError):
    # invalid shapes
    bloch.B([mat] * 2) * bloch.B([mat] * 3)

with pytest.raises(ValueError):
    # invalid shapes
    bloch.B.combine([bloch.B([mat] * 2), bloch.B([mat] * 3)])

"""
