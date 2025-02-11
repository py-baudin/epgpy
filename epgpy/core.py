"""EPG library

    Implemented Operators:
        * Operator: base operator class (does nothing)

        * MultiOperator: an operator made of a sequence of operators

        * NULL: convenience empty operator

        * ADC: Acquire signal. Required by `simulate`

        * SPOILER: perfect spoiler, sets transverse magnetization to 0

        * Wait(t): time offset operator (has no effect on computed values)

        * S(k): gradient operator

(todo)  * D(k, d): diffusion operator

        * T(alpha, phi): (ideal) RF-pulse operator

        * E(tau, T1, T2 [, g]): relaxation / precession operator

        * P(tau, g): precession operator

        * StateMatrix: container to store the phase states
            Defaults to equilibrium ([0, 0, 1])


    Making a sequence
    ===
        Just put them in a list.

        Example:
            spin_echo = [
                T(90, 90), S(1), E(10, 1000, 30),
                T(180, 0), S(1),  E(10, 1000, 30), ADC,
            ]

            # simulate signal
            signal = simulate(spin_echo)

    Functions:
    ===
        * simulate(sequence): sequence simulation

        * modify(sequence, ...): update sequence with object's parameters (T1, T2, ...)

        * getnshift(sequence): calculate the number of absolute phase
            shifts in the sequence

        * getshape(sequence): calculate the number of parameter dimensions

        * get_adc_times(sequence): calculate ADC opening times in the sequence


    Notes
    ===
        The phase state index 'k' is unit-less, and corresponds to the application of
        a unitary gradient pulse 'gt', where:

            ``` gt = gamma * G * tau ```

            gamma: gyro-magnetic ratio (kHz/T)
            G: gradient pulse amplitude (T/m)
            tau: gradient pulse duration (ms)

        To obtain the spacial frequency corrsponding to a k-state,
        simply multiply k with gt


    References
    ===
    * Weigel, 2014, Extended Phase Graphs: Dephasing, RF Pulses, and Echoes - Pure and Simple
    * Weigel et al., 2010, Extended Phase Graphs with Anisotropic Diffusion.


"""

from .utils import *
from .statematrix import StateMatrix
from .operators import *
from .functions import simulate, modify, get_adc_times, getshape, getnshift, imaging
