"""EPG diffusion operator and functions

based on:
> Weigel M, Schwenk S, Kiselev VG, Scheffler K, Hennig J:
  Extended phase graphs with anisotropic diffusion.
  Journal of Magnetic Resonance 2010; 205:276–285.

"""

import numpy as np
from . import operator, common, utils


class D(operator.Operator):
    """Anisotrophic diffusion operator

    D(tau, g, D, k=1)
    """

    # TODO:
    # combine adjacent D operators (add b matrices)

    def __init__(self, tau, D, k=None, *, method=None, name=None, duration=None):
        """initialize D operator

        tau: diffusion time (ms)
        D: diffusion matrix (mm^2/s)

        k: (rad/m) induced phase shift, if any
            If set, this operator must come right after an operator S(k)
        """

        tau, D, k = common.map_arrays((tau, D, k))
        self._shape, self._kdim = get_shape(tau, D, k)

        if name is None:  # default name
            name = common.repr_operator(
                "D", ["tau", "D", "k"], [tau, D, k], [".1f", "", ""]
            )

        self._duration = duration
        if duration is True:
            # match duration and tau if duration is True
            duration = tau

        self.tau = tau
        self.D = D
        self.k = k

        super().__init__(name=name, duration=duration)

    @property
    def shape(self):
        return self._shape

    @property
    def kdim(self):
        return self._kdim

    def _apply(self, sm):
        # compute b-matrix for L and T states
        xp = common.get_array_module()
        if self.k is None:
            bmatL = compute_bmatrix(self.tau, sm.k)
            bmatT = bmatL
        else:
            shift = xp.asarray(self.k * sm.kvalue)
            bmatL = compute_bmatrix(self.tau, sm.k)
            bmatT = compute_bmatrix(self.tau, sm.k - shift, sm.k)

        # get diffusion operator
        DL, DT = diffusion_operator(bmatL, bmatT, self.D)

        # apply
        sm.states[..., 0] = DT * sm.states[..., 0]
        sm.states[..., 2] = DL * sm.states[..., 2]
        sm.states[..., 1] = sm.states[..., ::-1, 0].conj()

        return sm


#
# functions


def compute_bmatrix(tau, k1, k2=None):
    """compute bmatrix for a linear change in k

    Args
        tau: diffusion time (ms)
        k1, k2: pre and post diffusion wavenumbers (rad/m)
    Returns
        bmat: b-matrix (s/mm^2)

    """
    xp = common.get_array_module()

    def outer(a, b):
        return a[..., xp.newaxis] * b[..., xp.newaxis, :]

    tau = tau * 1e-3  # ms to s

    k1 = xp.atleast_2d(k1) * 1e-3  # 1/m to 1/mm
    if k1.shape[-1] > 3:
        raise ValueError("Only 1d,2d and 3d wavenumbers are allowed")
    bmat = outer(k1, k1) * tau

    if k2 is None:
        return bmat

    # else
    k2 = xp.atleast_2d(k2) * 1e-3  # 1/m to 1/mm
    if k2.shape[-1] != k1.shape[-1]:
        raise ValueError("Incompatible numbers of dimensions for k1 and k2")
    kd = k2 - k1

    if xp.allclose(kd, 0):
        return bmat

    bmat = bmat + tau * (
        1 / 2 * outer(k1, kd) + 1 / 2 * outer(kd, k1) + 1 / 3 * outer(kd, kd)
    )
    return bmat


def diffusion_operator(bL, bT, D):
    """
    bL, bT: longitudinal and transveral b-matrices s/mm^2
    D: diffusion coefficient/tensor mm^2/s

    returns: EPG diffusion operator diagonal entries: exp(-bLD), exp(-bTD)
    """
    xp = common.get_array_module()
    if common.isscalar(D):
        # isotropic diffusion exp(-Tr(b)D)
        bL, bT = common.expand_arrays(bL, bT, append=False)
        idiag = xp.arange(bT.shape[-1])
        DL = xp.exp(-xp.sum(bL[..., idiag, idiag], axis=-1) * D)
        DT = xp.exp(-xp.sum(bT[..., idiag, idiag], axis=-1) * D)
    else:
        # anisotrophic diffusion exp(-Tr(bD))
        D = xp.asarray(D)
        bL, bT, D = common.expand_arrays(bL, bT, D, append=False)
        DL = xp.exp(-xp.sum(bL * D, axis=(-2, -1)))
        DT = xp.exp(-xp.sum(bT * D, axis=(-2, -1)))

    return DL, DT


#
# misc


def get_shape(tau, D, k):
    """check and get operator shape"""

    tau_shape = common.get_shape(tau)
    k_shape = common.get_shape(k)
    D_shape = common.get_shape(D)

    if not k_shape:
        k_shape = ()
    elif len(k_shape) == 1:
        k_shape = (1,) + k_shape

    if len(D_shape) == 1:
        raise ValueError("D can only be a scalar or a 2d matrix")
    elif len(set(D_shape[-2:])) == 2:
        raise ValueError("D must be a square 2d matrix")
    elif len(D_shape) and len(k_shape) and D_shape[-1] != k_shape[-1]:
        raise ValueError("Incompatible D and k dimensions")

    # shape is atleast (1,)
    shape = common.broadcast_shapes(tau_shape, D_shape[:-2], k_shape[:-1], [1])
    kdim = k_shape[-1] if k_shape else 1
    return shape, kdim
