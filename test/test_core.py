""" test epglib module """

import pytest
import itertools
import numpy as np
from epgpy import core, common


def test_hyperecho_sequence():
    """hyper-echo sequence (without relaxation)
    to test the code
    """

    # build sequence
    npulse = 201
    excit = core.T(90, 90)
    grad = core.S(1)
    adc = core.ADC
    pulse1 = core.T(10, 0)
    pulse2 = core.T(-10, 0)
    invert = core.T(180, 0)

    spinecho1 = [grad, pulse1, grad, adc]
    spinecho2 = [grad, pulse2, grad, adc]
    seq = [excit] + spinecho1 * npulse + [grad, invert, grad] + spinecho2 * npulse

    # run simulation
    sim = core.simulate(seq, probe="(F0, Z0)")

    # 'Hyper-echo signal must be F0=1, Z0=0 at the end
    assert not np.allclose(sim, [[1], [0]])
    assert np.allclose(sim[-1], [[1], [0]])
