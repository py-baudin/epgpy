# epgpy

## Description

A python library for simulating NMR signals using the Extended Phase Graph (EPG) formalism.

Its aims are:

- Simplicity (few dependencies, user-friendly, _pythonic_ syntax)
- Generality (can be used for simulations, optimization, etc.)
- Efficiency* (vectorization, GPU enabled)

*only some degree of efficiency can be expected from this implementation, if speed is really critical, other implementations may be more suitable.

An introductory tutorial is available [here](docs/basics.md).

A number of extensions are implemented:

- Arbitrary 3D gradients
- Isotropic and anisotropic diffusion
- Multi-compartment exchanges, including magnetization transfer
- Differentiability (e.g., for sequence optimization, cf. guides: [differentiation](docs/differentiation.md) and [sequence](docs/sequence.md))
- GPU compatible (via `cupy`)

A list of all available operators is given [here](docs/operators.md).
Please look into the `docs/` and `examples/` folders for tutorials and usage examples.

Disclaimer: this is a research project, and the authors give no guaranty on the validity of the generated results. 

## Example usage

Simulate a multi-spin-echo NMR sequence using EPG operators:

```python
from epgpy import epg

FA = 120 # flip angle (degrees)
ESP = 10 # echo spacing (ms)
Necho = 20 # num. echo

# relaxation times (multiple T2 values are simulated at once)
T1 = 150 # ms
T2 = [30, 40, 50] # ms

# operators
exc = epg.T(90, 90) # excitation
rfc = epg.T(FA, 0) # refocusing
rlx = epg.E(ESP/2, T1, T2) # relaxation
shift = epg.S(1, duration=ESP / 2) # spoiler
adc = epg.ADC # reading flag

# concatenate operators (in nested lists)
seq = [exc] + [[shift, rlx, rfc, shift, rlx, adc]] * Necho

# simulate the signal
signal = epg.simulate(seq)
# get adc times
times = epg.get_adc_times(seq)

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
![plot](docs/images/readme_mse_example.png)

## Publications

About `epgpy`:

- [ESMRMb 2025 poster (pdf)](docs/pdf/2025-10-10_esmrmb_poster_epgpy.pdf)

In use:

- [Vermeulen et al. « Quantitative Muscle Water  T2  Mapping Using RF Phase‐modulated 3D Gradient Echo Imaging ». Magnetic Resonance in Medicine (2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30545)
- [Slioussarenko et al. « A Steady‐state MR Fingerprinting Sequence Optimization Framework Applied to the Fast 3D Quantification of Fat Fraction and Water T1 in the Thigh Muscles ». Magnetic Resonance in Medicine (2025)](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.30490)
- [Slioussarenko et al. « Bi‐component Dictionary Matching for MR Fingerprinting for Efficient Quantification of Fat Fraction and Water  T1  in Skeletal Muscle ». Magnetic Resonance in Medicine (2024)](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.29901)

## References

These references were used for the implementation:

- Weigel, Matthias. « Extended Phase Graphs: Dephasing, RF Pulses, and Echoes - Pure and Simple: Extended Phase Graphs ». Journal of Magnetic Resonance Imaging (2015). https://doi.org/10.1002/jmri.24619.
- Weigel, M., S. Schwenk, V.G. Kiselev, K. Scheffler, et J. Hennig. « Extended Phase Graphs with Anisotropic Diffusion ». Journal of Magnetic Resonance (2010). https://doi.org/10.1016/j.jmr.2010.05.011.
- Gao, Xiang, Valerij G. Kiselev, Thomas Lange, Jürgen Hennig, et Maxim Zaitsev. « Three‐dimensional Spatially Resolved Phase Graph Framework ». Magnetic Resonance in Medicine (2021). https://doi.org/10.1002/mrm.28732.
- Malik, Shaihan J., Rui Pedro A.G. Teixeira, et Joseph V. Hajnal. « Extended Phase Graph Formalism for Systems with Magnetization Transfer and Exchange: EPG-X: Extended Phase Graphs With Exchange ». Magnetic Resonance in Medicine (2018). https://doi.org/10.1002/mrm.27040.
- Endres, Jonathan, Simon Weinmüller, Hoai Nam Dang, et Moritz Zaiss. « Phase Distribution Graphs for Fast, Differentiable, and Spatially Encoded Bloch Simulations of Arbitrary MRI Sequences ». Magnetic Resonance in Medicine 92, nᵒ 3 (2024): 1189‑1204. https://doi.org/10.1002/mrm.30055.


