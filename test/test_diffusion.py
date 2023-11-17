""" test epgpy.diffusion"""
import numpy as np
import pytest
from epgpy import operators, diffusion, statematrix


def test_compute_bmatrix():
    # scalar, 1ms, 2e3 rad/s -> 4e-3 s/mm^2
    bmat = diffusion.compute_bmatrix(1, 2e3)
    assert bmat.shape == (1, 1, 1)
    assert np.allclose(bmat, 4e-3)

    # k1=2e3, k2=3e3 -> b = 1e3 * (4 + 1/2 * 2 * 1 + 1/2 * 2 * 1 + 1/3)
    bmat = diffusion.compute_bmatrix(1, 2e3, 3e3)
    assert bmat.shape == (1, 1, 1)
    assert np.allclose(bmat, (4 + 2 + 1 / 3) * 1e-3)

    # 2d
    bmat = diffusion.compute_bmatrix(1, [2e3, 0])
    assert bmat.shape == (1, 2, 2)
    assert np.allclose(bmat, [[[4e-3, 0], [0, 0]]])

    bmat = diffusion.compute_bmatrix(1, [2e3, 3e3])
    assert bmat.shape == (1, 2, 2)
    assert np.allclose(bmat, [[[4e-3, 6e-3], [6e-3, 9e-3]]])

    # 3d
    bmat = diffusion.compute_bmatrix(1, [1e3, 2e3, 3e3])
    assert bmat.shape == (1, 3, 3)

    with pytest.raises(ValueError):
        bmat = diffusion.compute_bmatrix(1, [1e3, 2e3, 3e3, 4e3])
    with pytest.raises(ValueError):
        bmat = diffusion.compute_bmatrix(1, [1e3, 2e3], [1e3, 2e3, 3e3])

    # multi-dim
    bmat = diffusion.compute_bmatrix(1, [[1e3, 0], [0, 2e3], [1e3, 2e3]])
    assert bmat.shape == (3, 2, 2)
    assert np.allclose(bmat[0], [[1e-3, 0], [0, 0]])
    assert np.allclose(bmat[1], [[0, 0], [0, 4e-3]])
    assert np.allclose(bmat[2], [[1e-3, 2e-3], [2e-3, 4e-3]])

    # multi diom in shift
    bmat = diffusion.compute_bmatrix(1, [1e3, 0], [[1e3, 0], [0, 2e3], [1e3, 2e3]])
    assert bmat.shape == (3, 2, 2)


def test_diffusion_operator():
    # isotropic diffusion
    D = 1  # mm^2 / s
    bL = diffusion.compute_bmatrix(1, 1e3)  # 1e-3 s/mm^2
    bT = bL
    DL, DT = diffusion.diffusion_operator(bL, bT, D)
    assert DL.shape == DT.shape == (1,)  # one state
    assert np.allclose(DL, np.exp(-1e-3))

    bL = diffusion.compute_bmatrix(1, 1e3)  # 1e-3 s/mm^2
    bT = diffusion.compute_bmatrix(1, 1e3, 2e3)  # (2 + 1/3)e-3 s/mm^2
    DL, DT = diffusion.diffusion_operator(bL, bT, D)
    assert DL.shape == DT.shape == (1,)  # one state
    assert np.allclose(DL, np.exp(-1e-3))
    assert np.allclose(DT, np.exp(-(2 + 1 / 3) * 1e-3))

    # same but with non scalar wavenumbers
    bL = diffusion.compute_bmatrix(1, [1e3, 0, 0])
    bT = diffusion.compute_bmatrix(1, [1e3, 0, 0], [2e3, 0, 0])
    DL, DT = diffusion.diffusion_operator(bL, bT, D)
    assert DL.shape == DT.shape == (1,)  # one state
    assert np.allclose(DL, np.exp(-1e-3))
    assert np.allclose(DT, np.exp(-(2 + 1 / 3) * 1e-3))

    # multi-states
    bL = diffusion.compute_bmatrix(1, [1e3, 0])
    bT = diffusion.compute_bmatrix(1, [1e3, 0], [[1e3, 0], [2e3, 0], [3e3, 0]])
    DL, DT = diffusion.diffusion_operator(bL, bT, D)
    assert DL.shape == (1,)  # one state
    assert DT.shape == (3,)  # 3 states
    assert np.allclose(DL, np.exp(-1e-3))
    assert np.allclose(
        DT, [np.exp(-1e-3), np.exp(-(2 + 1 / 3) * 1e-3), np.exp(-(3 + 4 / 3) * 1e-3)]
    )

    # D matrix
    D = np.diag([1, 1, 1])  # mm^2 / s
    bL = diffusion.compute_bmatrix(1, [1e3, 0, 0])  # 1e-3 s/mm^2
    bT = bL
    DL, DT = diffusion.diffusion_operator(bL, bT, D)
    assert DL.shape == DT.shape == (1,)  # one state
    assert np.allclose(DL, np.exp(-1e-3))

    with pytest.raises(ValueError):
        # incompatible dimensions
        DL, DT = diffusion.diffusion_operator(bL, bT, np.diag([1, 1]))

    # anisotropic
    D = np.diag([2, 1, 0])  # mm^2 / s
    bL = diffusion.compute_bmatrix(1, [1e3, 0, 0])
    bT = diffusion.compute_bmatrix(1, [1e3, 0, 0], [1e3, 1e3, 0])
    DL, DT = diffusion.diffusion_operator(bL, bT, D)
    assert DL.shape == DT.shape == (1,)  # one state
    assert np.allclose(DL, np.exp(-np.trace(bL @ D, axis1=-2, axis2=-1)))
    assert np.allclose(DT, np.exp(-np.trace(bT @ D, axis1=-2, axis2=-1)))


def test_D_class():
    # unitary shift: 1e5 rad/s
    sm0 = statematrix.StateMatrix([1, 1, 0], kvalue=1e5)

    # no diffusion
    d1 = diffusion.D(1, 1e-3)  # tau=1ms, D=1e-3 mm2/s, k=0
    sm1 = d1(sm0)
    assert np.allclose(sm1.states, sm0.states)

    # instantaneous gradient
    shift1 = operators.S(1)
    shift2 = operators.S(-1)
    sm1 = shift2(d1(shift1(sm0)))
    att = np.exp(-sm1.kvalue ** 2 * d1.tau * d1.D * 1e-9)
    assert np.isclose(sm1.F0, att)

    # diffusion gradient
    d2 = diffusion.D(1, 1e-3, k=1)  # tau=1ms, D=1e-3 mm2/s, k=1
    sm1 = shift2(d2(shift1(sm0)))
    att = np.exp(-sm1.kvalue ** 2 * (1 / 4 + 1 / 12) * d1.tau * d1.D * 1e-9)
    assert np.isclose(sm1.F0, att)

    # spin echo experiment
    exc = operators.T(90, 90)
    ref = operators.T(180, 0)
    shift = operators.S(1)
    d1 = diffusion.D(1, 1e-3, k=1)
    d2 = diffusion.D(2e-1, 1e-3)
    ops = [exc, shift, d1, d2, ref, d2, shift, d1]

    sm0 = statematrix.StateMatrix(kvalue=1e4)  # kvalue in rad/m
    sm = sm0
    for op in ops:
        sm = op(sm)

    # expected attenuation
    D = d1.D * 1e-9  # diffusion in m^2/ms
    k = sm.kvalue
    att = np.exp(-2 / 3 * k ** 2 * d1.tau * D) * np.exp(-2 * k ** 2 * d2.tau * D)
    assert np.isclose(sm.F0, att)

    #
    # D is a matrix

    # 2d isotropic diffusion
    D = np.diag([1, 1])

    # spin echo experiment
    exc = operators.T(90, 90)
    ref = operators.T(180, 0)
    shift = operators.S([1, 0])  # 2d discrete gradient
    d1 = diffusion.D(1, D, k=[1, 0])
    d2 = diffusion.D(2e-1, D)
    ops = [exc, shift, d1, d2, ref, d2, shift, d1]
    sm = sm0
    for op in ops:
        sm = op(sm)

    # expected attenuation
    k = sm.kvalue
    D = 1e-9  # diffusion in m^2/ms
    att = np.exp(-2 / 3 * k ** 2 * d1.tau * D) * np.exp(-2 * k ** 2 * d2.tau * D)
    assert np.isclose(sm.F0, att)

    # 2d anisotropic diffusion
    D = np.diag([1, 2])

    # spin echo experiment
    exc = operators.T(90, 90)
    ref = operators.T(180, 0)
    shift = operators.S([1, 1])  # 2d discrete gradient
    d1 = diffusion.D(1, D, k=[1, 1])
    d2 = diffusion.D(2e-1, D)
    ops = [exc, shift, d1, d2, ref, d2, shift, d1]
    sm = sm0
    for op in ops:
        sm = op(sm)

    # expected attenuation
    k = sm.kvalue  # rad/m
    bT = sum(
        [
            diffusion.compute_bmatrix(1, [0, 0], [k, k]),
            diffusion.compute_bmatrix(2e-1, [k, k]),
            diffusion.compute_bmatrix(2e-1, [-k, -k]),
            diffusion.compute_bmatrix(1, [-k, -k], [0, 0]),
        ]
    )  # s/mm^2

    att = np.exp(-np.trace(bT @ D, axis1=-2, axis2=-1))
    assert np.isclose(sm.F0, att)
