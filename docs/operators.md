
# List of available operators

Following is a list of available operators and their arguments

## Transition (RF pulse)

Transition (instantaneous RF pulse) `T`
```python
# arguments: alpha (flip angle, degree), phi (phase, degree)
epg.T(alpha, phi) 
```

Phase offset operator `Phi`
```python
# arguments: phi (phase, degree)
epg.Phi(phi)
```

## Evolution (relaxation / precession)

Evolution (relaxation/precession operator) `E`
```python
# arguments: tau (time, ms), T1 (ms), T2 (ms), g (precession frequency, kHz) 
epg.E(tau, T1, T2, g=0)
```

Precession `P`
```python
# arguments: tau (ms), g (ms)
epg.P(tau, g)
```

Evolution (more general) `R`
```python
# arguments: rT (transverse evolution, a.u)
# rL (longitudinal evolution, a.u), r0 (longitudinal recovery, a.u) 
epg.R(rT=0, rL=0, *, r0=None)
```
Note: `epg.T(tau, T1, T2, g)` is equivalent to: `epg.R(rT=tau * (1/T2 + 2j * pi * g), rL=tau/T1, r0=tau/T1)` 

## Shift (phase states / gradients)

Shift (phase state shift) `S`
``` python
# arguments: k (wavenumber, rad/m)
# if `k` is an int, the standard 1-d a.u. EPG integer shift is applied
# if k is an array of int, an n-dim a.u. integer EPG shift is applied
# if k is (an array of) float, a n-dim rad/m gridded EPG shift is applied
epg.S(k)
```
Note: if `k` is 4-d, the 4th dimension if the accumulated time.

Gradient (phase state shift due to applied gradient) `G`
```python
# arguments: tau (time, ms), grad (gradient amplitude, mT/m)
epg.G(tau, grad)
```
Note: `epg.G(tau, grad)` is equivalent to `epg.S(k)` with `k = utils.get_wavenumber(tau, grad)`

Time accumulation (for T2* and B0 deviations) `C`
```python
# arguments: tau (time, ms)
epg.C(tau)
```
Note: works best in combination with the special ADC operator `epg.Imaging` and the setup operator `epg.System(modulation=-R2star + 1j*B0)`

## Diffusion

Diffusion (iso/anisotropic diffusion) `D`
``` python
# arguments: tau (ms), D (diffusion, mm^2/s), k (same as epg.S),
# if k!=None, always put right after `epg.S(k)` (with the same k value)
epg.D(tau, D, k=None)
```
Note: use scalar `D` for isotropic, and 3x3 ndarray `D` for anisotrophic diffusion. 

## Exchange 

# Exchange (n-dimensional exchange) `E`
``` python
# arguments: tau (time), khi (exchange rate, 1/ms), axis (int, exchange axis)
epg.X(tau, khi, *, axis=-1, T1=None, T2=None, g=None)
```
Note: `khi` can also be a NxN kinetic matrix. State matrix must have size 2 (or N) in set axis.

## Probes

# ADC
``` python
# returns F0 state of state matrix
epg.ADC
```

More featured ADC `Adc`
```python
# arguments: attr (state matrix attribute, str), phase (offset phase, degree)
# reduce (int, reduce axis: add up values along axes)
epg.Adc(attr='F0', phase=0, reduce=None)
```
Examples: `epg.Adc('Z0')`, `epg.Adc(phase=-phi)`

Custom ADC `Probe`
```python
# args: probe (str expression or callable),
# callable signature: `probe(sm, *args, **kwargs)`
# string expression: the state matrix attributes, numpy functions and kwargs are accessible
# args, kwargs: arguments and keyword arguments for the callable
epg.Probe(probe, *args, **kwargs)
```
Example: `epg.probe("F0.mean(axis=1)")`

DFT (discrete Fourier transform) `DFT`
```python
# arguments: coords (ndarray, voxel coordinates)
# assumes infinitely small voxel (ie. voxel_shape=`point` in epg.imaging)
epg.DFT(coords)
```

Imaging ADC `Imaging`
```python
# arguments: coords (ndarray, voxel coordinates), voxel_shape ('box', 'points'), voxel_size (m)
epg.Imaging(coords, voxel_shape='box', voxel_size=1)
```
Note: best used in combination with `epg.System(modulation=..., weights=...)`
cf. `utils.imaging` for more detailed description

Differentation: `Jacobian` and `Hessian` (cf. `differentiation.md`)
```python
# set variables of the Jacobian matrix: J[i, j] = d(signal[i]) / d(variables[j])
epg.Jacobian(variables)
# set variables of the Hessian tensor
# H[i, j, k] = d2(signal[i]) / d(variables1[j]) / d(variables2[k])
# if variables2 is None, it is set to variables1
epg.Hessian(variables1, variables2=None)
```
Note: differentiation must be activated in the requested operators  for the corresponding variables (eg.: `epg.E(tau, T1, T2, order1=['T1', 'T2'], order2=True))`
Typical use: `signal, jac, hess = epg.simulate(seq, probe=[ADC, Jacobian(...), Hessian(...)])`

## Utilities

Do-nothing operators
``` python
epg.NULL
epg.Wait(duration) # do nothing, but for a virtual duration
```

Ideal spoiler (cancel all transverse magnetization)
```python
epg.SPOILER
epg.Spoiler()
```

Reset operator (reset state matrix to equilibrium)
```python
epg.RESET
epg.Reset()
```


Proton density (set/change longitudinal equilibrium magnetization) `PD`
```python
# if reset==True, also reset current state matrix to new equilibrium
epg.PD(pd, reset=True)
```

System (set properties and attributes of the state matrix for use by other operators) `System`
```python
epg.System(...)
```
Example: `epg.Imaging` will depend on the following setup:
`epg.System(kvalue=..., tvalue=..., modulation=tau * (R2star + 2j*pi*B0), weights=weights)`
```
