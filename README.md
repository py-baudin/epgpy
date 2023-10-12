# epgpy
A python library for simulating NRM signals using the Extended Phase Graph (EPG) formalism.


## Example usage

Create an multi-spin echo NMR sequence using EPG operators, and simulate the resulting signal.

```python
from epgpy import operators, functions

FA = 120 # flip angle (degrees)
ESP = 10 # echo spacing (ms)
Nrf = 20 # num. rf

# relaxation times
T1 = 150 # ms
T2 = [30, 40, 50] # ms

# operators
exc = operators.T(90, 90) # excitation
rfc = operators.T(FA, 0) # refocussing
rlx = operators.E(ESP/2, T1, T2) # relaxation
shift = operators.S(1, duration=ESP / 2) # spoiler gradients
adc = operators.ADC # reading flag

# concatenate operators in nested lists
seq = [exc] + [[shift, rlx, rfc, shift, rlx, adc]] * Nrf

# simulate the signal
signal = functions.simulate(seq)
# get adc times
times = functions.get_adc_times(seq)

#
# plot signal
from matplotlib import pyplot as plt
plt.plot(times, plt.np.abs(signal))
plt.title('MSE signal decay')
plt.xlabel('time (ms)')
plt.ylabel('magnitude (a.u)')
plt.legend(labels=[f'{t2} ms' for t2 in T2], title='T2 values')
plt.show()

```
![plot](docs/readme_mse_example.png)
