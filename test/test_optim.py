import numpy as np
from epgpy import optim


def test_crlb():
    # dummy jacobian
    p = np.random.uniform(size=3)

    t = np.linspace(0, 1, 10)

    def signal(p):
        return np.cos(t * p[0]) + np.exp(t * (-p[1] + 1j * p[2]))

    def jac(p):
        exp = np.exp(t * (-p[1] + 1j * p[2]))
        return np.stack(
            [-t * np.sin(t * p[0]), -t * exp, t * 1j * exp],
            axis=1,
        )

    def hess(p):
        exp = np.exp(t * (-p[1] + 1j * p[2]))
        return np.stack(
            [
                np.stack([-(t ** 2) * np.cos(t * p[0]), t * 0, t * 0], axis=1),
                np.stack([t * 0, t ** 2 * exp, -1j * t ** 2 * exp], axis=1),
                np.stack([t * 0, -1j * t ** 2 * exp, -(t ** 2) * exp], axis=1),
            ],
            axis=2,
        )

    # check jac
    dp = np.random.uniform(size=len(p))
    fdiff = (signal(p + 1e-8 * dp) - signal(p)) * 1e8
    assert np.allclose(fdiff, jac(p) @ dp)

    # hessian
    fdiff = (jac(p + 1e-8 * dp) - jac(p)) * 1e8
    assert np.allclose(fdiff, hess(p) @ dp)

    hessp = hess(p) @ dp
    hessp = np.concatenate([hessp.real, hessp.imag], axis=0)

    # crlb
    cost, grad = optim.crlb(jac(p), H=hess(p))
    fdiff = (optim.crlb(jac(p + 1e-8 * dp)) - cost) * 1e8
    assert np.allclose(fdiff, grad @ dp)

    # with log10
    cost, grad = optim.crlb(jac(p), H=hess(p), log=True)
    fdiff = (optim.crlb(jac(p + 1e-8 * dp), log=True) - cost) * 1e8
    assert np.allclose(fdiff, grad @ dp)

    # crlb with weights
    weights = [1, 2, 3]
    cost, grad = optim.crlb(jac(p), H=hess(p), W=weights)
    fdiff = (optim.crlb(jac(p + 1e-8 * dp), W=weights) - cost) * 1e8
    assert np.allclose(fdiff, grad @ dp)
