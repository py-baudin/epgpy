""" EPG Transition operator and functions"""
import numpy as np
from . import common, opmatrix


class T(opmatrix.MatrixOp):
    """n-dimensional transition operator (instantaneous RF-pulse)"""

    def __init__(self, alpha, phi, *, axes=None, name=None, duration=None):
        """Init instantaneous RF-pulse operator

        Args:
            alpha: float
                flip angle (x axis rotation) (degree)
            phi: float
                phase (z-axis rotation) (degree)
            cf. Operator for other keyword arguments
        """
        params = common.map_arrays(alpha=alpha, phi=phi)

        if not name:
            # default name
            name = common.repr_operator(
                "T", ["alpha", "phi"], [alpha, phi], [".1f", "1f"]
            )

        # store parameters
        self.alpha = params["alpha"]
        self.phi = params["phi"]
        mat = rotation_operator(self.alpha, self.phi)

        # init operator
        super().__init__(mat, axes=axes, name=name, duration=duration)


# short cut classes
class Tx(T):
    def __init__(self, alpha, **kwargs):
        T.__init__(self, alpha, 0, **kwargs)


class Ty(T):
    def __init__(self, alpha, **kwargs):
        T.__init__(self, alpha, 90, **kwargs)


class Phi(opmatrix.MatrixOp):
    """Add phase offset"""

    def __init__(self, phi, *, axes=None, name=None):
        params = common.map_arrays(phi=phi)
        if not name:
            name = common.repr_operator("Phi", ["phi"], [phi], [".1f"])
        self.phi = params["phi"]
        mat = rotation_phi(self.phi)

        # init operator
        super().__init__(mat, axes=axes, name=name, duration=0)


# functions


def rotation_operator(alpha, phi):
    """rotation matrix (in degree)"""
    alpha, phi = common.expand_arrays(alpha, phi, append=True)
    return rotation_phi(phi) @ rotation_alpha(alpha) @ rotation_phi(-phi)


def rotation_alpha(alpha):
    """rotation matrix w/r to x axis"""
    xp = np

    a = xp.pi / 180.0 * xp.atleast_1d(alpha)

    mat = xp.ndarray(a.shape + (3, 3), dtype=xp.complex128)
    mat[..., 0, 0] = xp.cos(a / 2) ** 2
    mat[..., 0, 1] = xp.sin(a / 2) ** 2
    mat[..., 0, 2] = -1j * xp.sin(a)
    mat[..., 1, 0] = xp.sin(a / 2) ** 2
    mat[..., 1, 1] = xp.cos(a / 2) ** 2
    mat[..., 1, 2] = 1j * xp.sin(a)
    mat[..., 2, 0] = -1j / 2 * xp.sin(a)
    mat[..., 2, 1] = 1j / 2 * xp.sin(a)
    mat[..., 2, 2] = xp.cos(a)

    return mat


def rotation_phi(phi):
    """rotation matrix w/r to z axis"""
    xp = np

    p = xp.atleast_1d(phi) * xp.pi / 180.0

    mat = xp.zeros(p.shape + (3, 3), dtype=xp.complex128)
    mat[..., 0, 0] = xp.exp(1j * p)
    mat[..., 1, 1] = xp.exp(-1j * p)
    mat[..., 2, 2] = 1

    return mat


#
# diff

# 1st derivatives


def rotation_d_alpha(alpha, phi):
    """gradient of RF pulse w/r alpha"""
    return rotation_phi(phi) @ rotation_alpha_d(alpha) @ rotation_phi(-phi)


def rotation_d_phi(alpha, phi):
    """gradient of RF pulse w/r phi"""
    return rotation_phi_d(phi) @ rotation_alpha(alpha) @ rotation_phi(
        -phi
    ) - rotation_phi(phi) @ rotation_alpha(alpha) @ rotation_phi_d(-phi)


def rotation_alpha_d(alpha):
    xp = np

    a = xp.atleast_1d(alpha) * xp.pi / 180
    rot = xp.ndarray(a.shape + (3, 3), dtype=xp.complex128)
    rot[..., 0, 0] = -0.5 * xp.sin(a)
    rot[..., 0, 1] = 0.5 * xp.sin(a)
    rot[..., 0, 2] = -1j * xp.cos(a)
    rot[..., 1, 0] = 0.5 * xp.sin(a)
    rot[..., 1, 1] = -0.5 * xp.sin(a)
    rot[..., 1, 2] = 1j * xp.cos(a)
    rot[..., 2, 0] = -1j / 2 * xp.cos(a)
    rot[..., 2, 1] = 1j / 2 * xp.cos(a)
    rot[..., 2, 2] = -xp.sin(a)
    return rot * xp.pi / 180


def rotation_phi_d(phi):
    xp = np
    p = xp.atleast_1d(phi) * xp.pi / 180.0
    rot = xp.zeros(p.shape + (3, 3), dtype=xp.complex128)
    rot[..., 0, 0] = 1j * xp.exp(1j * p)
    rot[..., 1, 1] = -1j * xp.exp(-1j * p)
    rot[..., 2, 2] = 0
    return rot * xp.pi / 180


#
# 2d and cross derivatives


def rotation_d2_alpha(alpha, phi):
    """gradient of RF pulse w/r alpha"""
    return rotation_phi(phi) @ rotation_alpha_d2(alpha) @ rotation_phi(-phi)


def rotation_d2_alpha_phi(alpha, phi):
    """gradient of RF pulse w/r alpha"""
    return rotation_phi_d(phi) @ rotation_alpha_d(alpha) @ rotation_phi(
        -phi
    ) - rotation_phi(phi) @ rotation_alpha_d(alpha) @ rotation_phi_d(-phi)


def rotation_d2_phi(alpha, phi):
    """gradient of RF pulse w/r phi"""
    return (
        rotation_phi_d2(phi) @ rotation_alpha(alpha) @ rotation_phi(-phi)
        + rotation_phi(phi) @ rotation_alpha(alpha) @ rotation_phi_d2(-phi)
        - 2 * rotation_phi_d(phi) @ rotation_alpha(alpha) @ rotation_phi_d(-phi)
    )


def rotation_alpha_d2(alpha):
    xp = np
    a = xp.atleast_1d(alpha) * xp.pi / 180
    rot = xp.ndarray(a.shape + (3, 3), dtype=xp.complex128)
    rot[..., 0, 0] = -0.5 * xp.cos(a)
    rot[..., 0, 1] = 0.5 * xp.cos(a)
    rot[..., 0, 2] = 1j * xp.sin(a)
    rot[..., 1, 0] = 0.5 * xp.cos(a)
    rot[..., 1, 1] = -0.5 * xp.cos(a)
    rot[..., 1, 2] = -1j * xp.sin(a)
    rot[..., 2, 0] = 1j / 2 * xp.sin(a)
    rot[..., 2, 1] = -1j / 2 * xp.sin(a)
    rot[..., 2, 2] = -xp.cos(a)
    return rot * (xp.pi / 180) ** 2


def rotation_phi_d2(phi):
    xp = np
    p = xp.atleast_1d(phi) * xp.pi / 180.0
    rot = xp.zeros(p.shape + (3, 3), dtype=xp.complex128)
    rot[..., 0, 0] = -xp.exp(1j * p)
    rot[..., 1, 1] = -xp.exp(-1j * p)
    rot[..., 2, 2] = 0
    return rot * (xp.pi / 180) ** 2
