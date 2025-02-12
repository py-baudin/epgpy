# The `epg.sequence` module

## Introduction

The `epg.sequence` module provides a simplified syntax to define EPG sequences and
simulate the resulting signal and its derivatives.

```python
from epgpy.sequence import Sequence, operators
T, E, S = operators.T, operators.E, operators.S

# define a sequence using operators and variables
seq = Sequence([T(90,90)] + [S(1), E(5,'T1','T2'), T(180, 0), S(1), E(5, 'T1', 'T2'), 'ADC'] * 10)

seq.variables
# -> {'T1', 'T2'}

# simulate the sequence by setting the variable's values
seq(T1=1.4e3, T2=3e1)
# -> array([[0.90483742, 0.81873075, 0.74081822, 0.67032005, 0.60653066, 0.54881164, 0.4965853 , 0.44932896, 0.40656966, 0.36787944]])
```


New objects are defined:

- `Variable`: a variable
- `operators.T/E/S/...`: virtual operators, dependent on some variables and constants
- `Sequence`: a sequence, comprised of virtual operators

Sequence building is done in 3 steps:

1. define the variables (e.g. `T2`, `flip-angle`, etc.),
2. setup the operators using the variables,
3. create a sequence from the list of operators.

Most `epgpy` operators (i.e. in  `epgpy.operators`) exist as virtual operators (in `epgpy.sequence.operators`).

```python
from epg.sequence import operators

# Transition operator (instantaneous rf-pulse)
# arguments: flip-angle (degree), phase (degree)
operators.T(alpha, phi)

# evolution operator (relaxation and precession)
# arguments: duration (ms), T1 (ms), T2 (ms), precession (kHz)
operators.E(tau, T1, T2, g=freq)

# shift operator (unitary 1d gradient)
# arguments: (integer) phase state increment (a.u. or rad/m)
operators.S(k)

# utilities
adc = operators.ADC # enable recording the current phase state
spoiler = operators.SPOILER # perfect gradient spoiler
wait = operators.Wait(1.) # do nothing for some time (ms)

# etc.
```

## Sequence building

### Variables and operators

Operator's variables can be defined explicitly using the `Variable` class, or
implicitly, by passing string arguments to operators.

```python
from epg.sequence import Variable, Sequence, operators

tau = Variable("tau") # explicit variable definition
T1 = 1000 # constant
T2 = "T2" # implicit T2 variable definition
rlx = operators.E(tau, T1, T2) 

# set of variables for operator `relax`
assert rlx.variables == {"tau", "T2"}

```

Operator's variables can be replaced by constants by passing a value as keyword:

```python
rlx2 = rlx(tau=5)
assert rlx2.variables == {"T2"}
```

Operators can be parameterized by variable-dependent expressions:

```python
b1 = Variable("b1") # B1 attenuation factor
rf = operators.T(90 * b1, 90)
assert rf.variables == {"b1"}
```

### Sequence

A sequence is defined by a list of operators.
An `ADC` object is required to tell the simulation when to acquire the signal.

```python
necho = 10
adc = operators.ADC
seq = Sequence([rf, rlx, adc] * necho)
```

Variables of the sequence's operators are variables of the sequence.
```python
assert seq.variables == {"b1", "tau", "T2"}
```

The sequence object is used to simulate the signal.
The outputs of the simulation are arrays whose shape depend on the number of ADC.

```python
signal = seq(b1=1, tau=5, T2=30)
assert signal.shape == (1, 10) # 1x dimension, 10x ADC
```

The result from multiple variable values can be obtained in a single call,
by passing arrays to the variable:

```python
signal = seq(b1=1, tau=5, T2=[30, 40, 50])
assert signal.shape == (3, 10) # 3x T2 values, 10x ADC
```

Various simulation functions are available: 

```python
# simple wrapper of the epgpy.function.simulate function
signal = seq.simulate({'b1': '0.8', 'tau': '5', 'T2': '30'})
assert signal.shape == (10, 1) # 10x ADC, 1x dimension

# signal simulation
signal = seq(b1=0.8, tau=5, T2=30) # or
signal = seq.signal(b1=0.8, tau=5, T2=30)
assert signal.shape == (1, 10) # 1x dimension, 10x ADC

# Jacobian matrix for selected variables
signal, jac = seq.jacobian(["T2", "b1"])(b1=0.8, tau=5, T2=30)

# Hessian matrix (tensor) for selected variables
signal, grad, hess = seq.hessian(["magnitude", "T2", "b1"])(b1=0.8, tau=5, T2=30)

# CRLB (sequence optimization objective function)
seq.crlb(['T2', 'b1'])(b1=0.9, T1=1000, T2=30)

# Confidence intervals
seq.confint(obs, ['T2', 'b1'])(b1=0.9, T1=1000, T2=30)


```


### Tips

```python
# Operator's variables can be passed directly as string
seq = Sequence([E('tau', T1, T2), T('alpha', phi), ...])

# ADC, SPOILER, RESET operators can be passed as string
`Sequence([op1, op2, ..., 'ADC', 'SPOILER'])

# keyword options to epg.functions.simulate can be set at different places:
seq = Sequence(ops, options={'max_nstate': 10, 'disp': True})
seq.signal(options={'max_nstate': 10, 'disp': True})(...)

# Calling `Sequence.signal`, `.jacobian`, `.hessian`, `.crlb` or `.confint`
# without passing the variable's values returns a function
# with arguments: (values_dict=None, **values). For instance:
seq.crlb(variables1, gradient=variables2, **options)({var1: value1}, var2=value2)
```


## Examples

### Multi-spin echo simulation (MSE)

Make sure to reuse operators for efficiency.

```python
# operators
exc = operators.T(90, 90)
inv = operators.T(180, 0)
spl = operators.S(1, duration=5)
rlx = operators.E(5, 1645, "T2")

# MSE
necho = 17
seq = Sequence([exc] + [spl, rlx, inv, spl, rlx, "ADC"] * necho)

# simulate signal
times, signal = seq.simulate({'T2': 50}, adc_time=True)
```


### Spoiled gradient echo (SPGR)

Nested lists can be used for more compact definitions.

```python
# operators
rf = operators.T(14.8, "phi")
spl = operators.S(1, duration=5)
rlx = operators.E(5, 1645, "T2")
adc = operators.Adc(phase='phi') # ADC with phase compensation

# SPGR
necho = 400
phases = 58.5 * np.arange(necho)**2
seq = Sequence([[rf(phi=phase), rlx, adc(phi=-phase), rlx, spl] for phase in phases])

# simulate signal
times, signal = seq.simulate({'T2': 50}, adc_time=True)
```


### Double echo in steady state (DESS)

The same variables can be shared by different operators.

```python
# operators
TR, TE = 19.9, 4.2
rf = operators.T(45, 0)
spl = operators.S(1)
rlx1 = operators.E(TE, 800, "T2", duration=TE)
rlx2 = operators.E(TR - 2*TE, 800, "T2", duration=TR - 2*TE)

# sequence definition
necho = 200
seq = Sequence([rf, rlx1, "ADC", spl, rlx2, "ADC", rlx1] * necho)

# simulate signal
signal = seq.signal(T2=70)
```



## Differentiation

In some situations, one may need to compute the derivatives of the signal with respect to some of the variables.
Examples: sequence optimization, confidence interval calculation.

This is done using the `gradient` and `hessian` functions of the sequence object.

```python
signal = seq.signal(b1=0.8, tau=5, T2=30)
assert hess.shape == (1, 10)
signal, jac = seq.gradient(["T2", "b1"])(b1=0.8, tau=5, T2=30)
assert hess.shape == (1, 10, 2)
signal, grad, hess = seq.hessian(["T2", "b1"])(b1=0.8, tau=5, T2=30)
assert hess.shape == (1, 10, 2, 2)

# to obtain the partial derivatives of the signal with respect to the magnitude, `magnitude` can be added to the variable list:
# `magnitude` is available for seq.gradient, seq.hessian and seq.crlb
signal, grad, hess = seq.hessian(["magnitude", "T2", "b1"], b1=0.8, tau=5, T2=30)
assert hess.shape == (1, 10, 3, 3)
```


When dealing with 2nd derivatives, avoid computing unnecessary partial derivatives in the Hessian tensors 
by passing different variables for the "rows" (axis=-2) and the "columns" (axis=-1):

```python
signal, grad, hess = seq.hessian(["magnitude", "T2"], ["tau"])(b1=0.8, tau=5, T2=30)
# 2 row variables, 1 column variable
assert hess.shape == (1, 10, 2, 1)
```

The CRLB of the signal, and its gradient are also available:

```python
# compute the CRLB (internally calling seq.gradient)
crlb = seq.crlb(["T2", "b1"])(b1=0.8, tau=5, T2=[30, 40, 50])
assert crlb.shape == (3,) # 3 T2 values

# compute the CRLB and its gradient (internally calling seq.hessian)
crlb, dcrlb = seq.crlb(["T2", "b1"], gradient=["tau"])(b1=0.8, tau=5, T2=[30, 40, 50])
assert dcrlb.shape == (3, 1) # 3 T2 values, 1 gradient variable
```

