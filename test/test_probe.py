import numpy as np
import pytest
from epgpy import common, statematrix, probe, utils

StateMatrix = statematrix.StateMatrix
Probe = probe.Probe
Adc = probe.Adc
ADC = probe.ADC


def test_probe_class():
    xp = common.get_array_module()

    sm = StateMatrix([1, 1, 0.5], nstate=1)
    probe = Probe(lambda sm: sm.F0)
    assert probe(sm, inplace=True) is sm
    assert np.allclose(probe.acquire(sm), [1])

    probe = Probe(lambda sm: sm.F)
    acq1 = probe.acquire(sm)
    assert np.allclose(acq1, [0, 1, 0])

    # update state matrix
    sm.states[:, 0] = xp.array([0.5, 0.5, 0])
    sm.states[:, 2] = xp.array([0.5, 0.5, 0])
    acq2 = probe.acquire(sm)
    assert np.allclose(acq1, [0, 1, 0])  # unchanged
    assert np.allclose(acq2, [0.5, 1, 0.5])

    # using eval and axes names
    sm = StateMatrix([1, 1, 0.5], nstate=1, shape=(3, 2))
    assert np.allclose(Probe("F").acquire(sm), sm.F)
    assert np.allclose(Probe("F0").acquire(sm), sm.F0)
    assert np.allclose(Probe("Z").acquire(sm), sm.Z)
    assert np.allclose(Probe("Z0").acquire(sm), sm.Z0)

    # load axes in namespace
    axes = utils.Axes("T2", "B1")
    probe = Probe("F0.mean(axes.T2)", axes=axes)
    sm = StateMatrix(shape=(3, 4))
    assert sm.F0.shape == (3, 4)
    assert probe.acquire(sm).shape == (4,)  # reduced


def test_adc_class():
    sm = StateMatrix([[[1j, -1j, 0.5]], [[-1j, 1j, 0.5]]])
    adc = Adc()
    assert np.allclose(adc.acquire(sm), [1j, -1j])

    adc = Adc(phase=90)  # phase correction
    assert np.allclose(adc.acquire(sm), [-1, 1])
    assert np.allclose(adc.post(ADC.acquire(sm)), [-1, 1])  # equivalent

    adc = Adc(phase=90, weights=[2, 0.5], reduce=0)  # reduce
    assert np.allclose(adc.acquire(sm), -1.5)

    adc = Adc(phase=90, weights=1)  # reduce (implicit)
    assert np.allclose(adc.acquire(sm), 0)

    adc = Adc(phase=90, weights=[2, 0.5], reduce=...)  # no reducing
    assert np.allclose(adc.acquire(sm), [-2, 0.5])

    with pytest.raises(ValueError):
        Adc(weights=[2, 0.5], reduce=1)  # wrong dimension
