import numpy as np
from epgpy import epg

#  main objects

# state matrix
sm = epg.StateMatrix()

# attributes/properties
sm.states  # (ndarray) coefficients of the phase states
sm.nstate  # number of EPG phase states
sm.shape  # shape of the jointly simulated states
sm.equilibrium  # (ndarray) coefficients of the equilibrium phase states

# initialization options
epg.StateMatrix([1, 1, 0])  # initial state != equilibrium
epg.StateMatrix(equilibrium=[0, 0, 10])  # different equilibrium
epg.StateMatrix(density=10)  # like above
epg.StateMatrix(max_nstate=10)  # cap on phase states number

# wavenumbers
sm = epg.StateMatrix(kvalue=1)  # set unitary wavenumber to 1.0 rad/m
sm.k  # wavenumber (value of phase states)

#
# base operators

# transition (ideal rf-pulse)
# arguments: flip-angle (degree), phase (degree)
inversion = epg.T(180, 0)

# evolution (relaxation, precession)
# arguments: duration (ms), T1 (ms), T2 (ms), precession (kHz)
relax = epg.E(5, 1645, 50, g=0)

# shift (unit-less gradient)
# arguments: phase state increment (integer)
shift = epg.S(1)

# common arguments
# set display name
excit = epg.T(90, 90, name="excit")
# set virtual duration (ms): this has no effect on the simulation
shift = epg.S(1, duration=5)

# utilities
adc = epg.ADC  # enable recording phase state
probe = epg.Probe("Z0")  # custom recording function
spoiler = epg.SPOILER  # perfect gradient spoiler
wait = epg.Wait(1.0)  # does nothing for some time (ms)


#
# applying operators

""" One can apply the operators directly onto the state matrix:"""

# initial states
sm0 = epg.StateMatrix()

# rf pulse:
sm1 = excit(sm0)  # mix transverse and longitudinal states

# shift
sm2 = shift(sm1)  # shift transverse states (append new states if needed)

# relaxation
sm3 = relax(sm2)  # apply signal decay and precession


#
# sequence definition and functions

"""
It is more convenient to define a sequence (a list of operators) and use the `simulate` function.
"""

# define a sequence
spinecho = [excit, shift, relax, inversion, shift, relax, adc]

# simulate signal
signal = epg.simulate(spinecho)

# calculate timing, **based on the "duration" attribute**
echo_time = epg.get_adc_times(spinecho)


#
# examples

# multi-spin echo (17 TE)
# note: make sure to reuse operators for efficiency
necho = 17
mse = [excit] + [shift, relax, inversion, shift, relax, adc] * necho

# spoiled gradient echo
# note: nested lists can be used for more compact definitions
# note: the operateur "epg.Adc" accepts a `phase` argument to compensate for the RF pulse phase
necho = 400
phases = 58.5 * np.arange(necho) ** 2
spgr = [[epg.T(14.8, phase), relax, epg.Adc(phase=-phase), relax, shift] for phase in phases]


# Double echo in steady state (DESS)
necho = 200
TR, TE = 19.9, 4.2
rf = epg.T(45, 0)
relax1 = epg.E(TE, 800, 70, duration=True)
relax2 = epg.E(TR - 2 * TE, 800, 70, duration=True)
dess = [rf, relax1, adc, shift, relax2, adc, relax1] * necho


#
# plotting

import numpy as np
from matplotlib import pyplot as plt

mse_times, mse_signal = epg.simulate(mse, adc_time=True)
plt.figure("mse")
mag = plt.plot(mse_times, np.abs(mse_signal))
plt.title("MSE simulation")
plt.xlabel("time (ms)")
plt.ylabel("magnitude")
plt.twinx()
pha = plt.plot(mse_times, np.angle(mse_signal), ":")
plt.ylabel("phase (rad)")
plt.legend([mag[0], pha[0]], ["magnitude", "phase"])
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.tight_layout()

spgr_times, spgr_signal = epg.simulate(spgr, adc_time=True)
plt.figure("spgr")
mag = plt.plot(spgr_times, np.abs(spgr_signal))
plt.title("SPGR simulation")
plt.xlabel("time (ms)")
plt.ylabel("magnitude")
plt.ylim(0, plt.ylim()[1])
plt.twinx()
pha = plt.plot(spgr_times, np.angle(spgr_signal), ":")
plt.ylabel("phase (rad)")
plt.legend([mag[0], pha[0]], ["magnitude", "phase"])
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.tight_layout()


dess_times, dess_signal = epg.simulate(dess, adc_time=True)
plt.figure("dess")
mag1 = plt.plot(dess_times[::2], np.abs(dess_signal)[::2])
mag2 = plt.plot(dess_times[1::2], np.abs(dess_signal)[1::2])
plt.title("DESS simulation")
plt.xlabel("time (ms)")
plt.ylabel("magnitude")
plt.ylim(0, plt.ylim()[1])
plt.twinx()
pha1 = plt.plot(dess_times[::2], np.angle(dess_signal)[::2], ":")
pha2 = plt.plot(dess_times[1::2], np.angle(dess_signal)[1::2], ":")
plt.ylabel("phase (rad)")
plt.legend(
    [mag1[0], pha1[0], mag2[0], pha2[0]],
    ["magnitude-1", "phase-1", "magnitude-2", "phase-2"],
)
plt.ylim([-np.pi, np.pi])
plt.grid()
plt.tight_layout()


plt.show()
