""" test core.pulse module """

import numpy as np
import pytest
from epgpy import core as epg, rfpulse


RFPulse = rfpulse.RFPulse
estimate_rf = rfpulse.estimate_rf


def test_rfpulse_class():
    """test RFPulse class"""

    # dummy pulse
    duration = 10  # ms
    values = np.linspace(0, 1, 100)

    with pytest.raises(ValueError):
        # no alpha or rf
        RFPulse(values, duration)

    # basic pulse
    pulse = RFPulse(values, duration, rf=1)
    assert len(pulse.operators) == 100

    # check generated sequence
    pulse = RFPulse([1, 1j], duration=1, rf=1, T1=1000, T2=100)
    assert [type(op) for op in pulse.operators] == [epg.T, epg.E, epg.T, epg.E]
    assert [op.duration for op in pulse.operators] == [0.5, 0, 0.5, 0]
    assert pulse.operators[0].alpha == 180
    assert pulse.operators[0].phi == 0
    assert pulse.operators[2].alpha == 180
    assert pulse.operators[2].phi == 90

    # check RFPulse sequence using dummy values
    sm0 = epg.StateMatrix()
    pulse = RFPulse(values=[0.5], duration=1, rf=1)
    sm = pulse(sm0)
    # assert np.all(np.isclose(sm.states[0, 0, :], [0, 0, -1]))  # full inversion
    assert np.isclose(pulse.alpha, 90)
    assert np.all(np.isclose(sm.states[0, 0], [-1j, 1j, 0]))  # 90 degree

    pulse = RFPulse(values=[0.5, 0.5], duration=1, rf=1)
    assert len(pulse.operators) == 2
    sm = pulse(sm0)
    assert np.all(np.isclose(sm.states[0, 0, :], [0, 0, -1]))  # full inversion

    pulse = RFPulse(values=[0.25, 0.5j], duration=1, rf=1)
    assert len(pulse.operators) == 2
    sm = pulse(sm0)
    assert np.all(np.isclose(sm.states[0, 0, 0], np.sqrt(1 / 2) - 1j * np.sqrt(1 / 2)))


def test_estimate_rf():
    """ """

    npoint = 50
    mags = np.sin(np.linspace(0, np.pi, npoint))
    phases = np.sin(np.linspace(0, np.pi)) * np.pi
    values = mags * np.exp(1j * phases)
    duration = 1  # ms

    # optimize values
    target_alpha = 100
    signal_trans = np.sin(target_alpha / 180 * np.pi)
    signal_long = np.cos(target_alpha / 180 * np.pi)
    rf = estimate_rf(values, target_alpha)

    # test rf
    op = RFPulse(values, duration, rf=rf)
    signal = op(epg.StateMatrix()).states

    assert np.isclose(signal_trans, np.abs(signal[0, 0, 0]))
    assert np.isclose(signal_long, np.real(signal[0, 0, 2]))


# @pytest.mark.skip(reason="slice profile not validated")
# def test_slice_profile():
#     """test profile simulation"""

#     npoint = 501
#     duration = 5  # ms
#     times = np.linspace(-duration / 2, duration / 2, npoint)  # ms

#     # make sinc pulse
#     freq = 5  # kHz
#     pulse_values = np.sinc(2 * np.pi * freq * times)
#     pulse = RFPulse(pulse_values, duration, alpha=90)

#     # sampling frequency (kHz)
#     maxfreq = npoint / duration / 2
#     # convert to spatial position
#     gradient = 100
#     maxpos = rfpulse.freq_to_space(maxfreq * 2, gradient)
#     positions = maxpos * np.linspace(-0.5, 0.5, npoint)

#     # calculate profiles
#     profile1 = rfpulse.slice_profile(pulse, gradient, positions)

#     # Fourier transform of spectrum
#     fft = np.fft.fftshift(np.fft.fft(pulse_values))

#     # assert bandwidth are the same
#     mask_ref = np.abs(fft) > 0.5 * np.max(np.abs(fft))
#     mask_sig1 = np.abs(profile1) > 0.5 * np.max(np.abs(profile1))
#     assert np.mean(mask_ref != mask_sig1) < 1e-5
