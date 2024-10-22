
# List of available operators

Following is a list of available operators and their arguments

## Transition (RF pulse)

Transition (RF pulse)
```python
# arguments: alpha (flip angle, degree), phi (phase, degree)
epg.T(alpha, phi) 
```

Phase offset operator
```python

# arguments: phi (phase, degree)
epg.Phi(phi)
```

## Evolution (relaxation / precession)

```python
# Evolution (relaxation/precession operator)
# arguments: tau (time, ms), T1 (ms), T2 (ms), g (precession frequency, kHz) 
epg.E(tau, T1, T2, g=0)

# Precession
# arguments: tau (ms), g (ms)
epg.P(tau, g)

# Evolution (more general)
# arguments: rT (transverse evolution, a.u)
# rL (longitudinal evolution, a.u), r0 (longitudinal recovery, a.u) 
epg.R(rT=0, rL=0, *, r0=None)
```
Note: `epg.T(tau, T1, T2, g)` is equivalent to: `epg.R(rT=tau * (1/T2 + 2j * pi * g), rL=tau/T1, r0=tau/T1)` 

## Shift (phase states / gradients)

``` python
# Shift
# arguments: k (wavenumber, rad/m)
# note: if k is an int, the standard 1-d a.u. EPG integer shift is applied
# if k is an array of int, an n-dim a.u. integer EPG shift is applied
# if k is (an array of) float, a n-dim rad/m gridded EPG shift is applied
epg.S(k)
```
Note: if `k` is 4-d, the 4th dimension if the accumulated time.

```python
# Gradient
# arguments: tau (time, ms), grad (gradient amplitude, mT/m)
epg.G(tau, grad)
```
Note: `epg.G(tau, grad)` is equivalent to `epg.S(k)` with `k = utils.get_wavenumber(tau, grad)`

```python
# Time accumulation (for T2* and B0 deviations)
# arguments: tau (time, ms)
epg.C(tau)
```
Note: works best in combination with the special ADC operator `epg.Imaging` and the setup operator `epg.System(modulation=-R2star + 1j*B0)`

## Diffusion

``` python
# Diffusion (iso/anisotropic diffusion)
# arguments: tau (ms), D (diffusion, mm^2/s), k (same as epg.S), 
epg.D(tau, D, k=None)
# note: use scalar D for isotropic, and 3x3 ndarray D for anisotrophic diffusion
# note: if k!=None, always put right after `epg.S(k)` (with the same k value)
```

## Exchange 
``` python
# Exchange (n-dimensional exchange) 
# arguments: tau (time), khi (exchange rate, 1/ms), axis (int, exchange axis)
epg.X(tau, khi, *, axis=-1, T1=None, T2=None, g=None)
# note: khi can also be a NxN kinetic matrix
# note: state matrix must have size 2 (or N) in set axis
```

## ADC

``` python
# ADC
# returns F0 state of state matrix
epg.ADC

# more featured ADC
# arguments: attr (state matrix attribute, str), phase (offset phase, degree)
# reduce (int, reduce axis: add up values along axes)
epg.Adc(attr='F0', phase=0, reduce=None)
# examples: `epg.Adc('Z0')`, `epg.Adc(phase=-phi)`

# custom ADC
# args: probe (str expression, callable),
# args, kwargs: arguments and keyword arguments for the callable
epg.Probe(probe, *args, **kwargs)
# callable signature: `probe(sm, *args, **kwargs)`
# string expression: the state matrix attributes, numpy functions and kwargs are accessible
# example: `epg.probe("F0.mean(axis=1)")`

# DFT (discrete Fourier transform)
# arguments: coords (ndarray, voxel coordinates)
# assumes infinitely small voxel (ie. voxel_shape=`point` in epg.imaging)
epg.DFT(coords)

# Imaging (imaging ADC)
# arguments: coords (ndarray, voxel coordinates), voxel_shape ('box', 'points'), voxel_size (m)
epg.Imaging(coords, voxel_shape='box', voxel_size=1)
# note: best used in combination with epg.System(modulation=..., weights=...)
# cf. `utils.imaging` for more detailed description

# Jacobian and Hessian probes (cf. `differentiation.md`)
# set variables of the Jacobian matrix: J[i, j] = d(signal[i]) / d(variables[j])
epg.Jacobian(variables)
# set variables of the Hessian tensor
# H[i, j, k] = d2(signal[i]) / d(variables1[j]) / d(variables2[k])
# if variables2 is None, it is set to variables1
epg.Hessian(variables1, variables2=None)
# note: differentiation must be activated in the requested operators 
# for the corresponding variables (eg.: `epg.E(tau, T1, T2, order1=['T1', 'T2'], order2=True))`
# typical use: `signal, jac, hess = epg.simulate(seq, probe=[ADC, Jacobian(...), Hessian(...)])`
```


## Utilities

``` python
# do-nothing operators
epg.NULL
epg.Wait(duration) # do nothing, but for a virtual duration

# ideal spoiler, cancel all transverse magnetization 
epg.SPOILER
epg.Spoiler()

# reset operator, reset state matrix to equilibrium
epg.RESET
epg.Reset()

#
# setup 

# proton density, set/change longitudinal equilibrium magnetization
# if reset==True, also reset current state matrix to new equilibrium
epg.PD(pd, reset=True)

# system
# set properties and attributes of the state matrix for use by other operators
# example, `epg.Imaging` will depend on the following setup:
# epg.System(kvalue=..., tvalue=..., modulation=tau * (R2star + 2j*pi*B0), weights=weights)`
epg.System(...)
```
