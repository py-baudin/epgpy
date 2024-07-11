""" Provide Sequence building class

# variable definition
T1 = Variable("T1")
T2 = Variable("T2")
b1 = Variable("b1", dtype=float)

# define a sequence
ops = [T(90 * b1, 90), S(1), E(9.5, T1, T2), ...] # `b1` is a variable
seq = Sequence(ops)

ops[0].variables --> {"b1"} # operator's variables
ops[1].variables --> {"T1", "T2"}
ops[0](b1=0.9) --> epg.T(0.1*90, 90) # operator instantiation

# base functions
seq.variables --> {"b1", "T1", "T2"} # sequence's variables
seq.build(b1=0.9, T1=1000, T2=30) --> [epg.T(0.1*90, 90), ...] # sequence instantiation
seq.simulate(T2=30, ...) --> value  # sequence simulation

# helper functions
signal = seq.signal(...)
signal, gradient = seq.gradient(variables, ...)
sig, grad, hess = seq.hessian(variables, ...)

# Tips

# Operator's variables can be given as string
seq = Sequence([E('tau', T1, T2), T('alpha', phi), ...])

# ADC flag and SPOILER operator can be passed as string
`Sequence([op1, op2, ..., 'ADC', 'SPOILER'])

# partial Hessian (avoid computing unnecessary partial derivatives)
seq.hessian([var1, var3], [var2, var3]) # different variables in rows and columns
seq.crlb([var1, var2], gradient=["var3"]) # gradient can be True/False or a list of variables

"""

import abc
import types
import inspect
import numpy as np
from . import operators as base, functions, diff, common, optim, rfpulse


class Sequence:
    """Virtual sequence"""

    def __init__(self, ops, *, name=None, **options):
        """Build virtual sequence from a list of virtual operators

        Args:
            ops: list of operators
            name: sequence's name
            **options: `simulate` keyword options
        """
        self.operators = []
        self.variables = set()
        self.options = options  # simulate options
        self.name = name  # sequence name
        self.extend(ops)

    def __len__(self):
        return len(self.operators)

    def __repr__(self):
        return self.name if self.name else f"Sequence({len(self)})"

    def __call__(self, *args, **kwargs):
        """Alias for "simulate" """
        return self.simulate(*args, **kwargs)

    def __getitem__(self, i):
        """return i-th operator"""
        return self.operators[i]

    def __iter__(self):
        return iter(self.operators)

    def extend(self, ops):
        """Append operators to the sequence"""
        ops = flatten_list(ops)
        for op in ops:
            if isinstance(op, str):
                # parse string operator
                op = self.parse_string_operator(op)
            elif isinstance(op, Sequence):
                # update simulate options
                self.options = {**op.options, **self.options}
            elif not isinstance(op, VirtualOperator):
                raise TypeError(f"Invalid VirtualOperator: {op}")

            self.variables |= getattr(op, "variables", set())
            self.operators.append(op)

    def flatten(self):
        """return list of virtual operators, including operators from sub-Sequences"""
        return [op for _op in self.operators for op in _op.flatten()]

    def build(self, **kwargs):
        """Return list of EPG Operators

        Args:
            **kwargs: variables' values
        """
        # flat virtual operator's list
        operators = self.flatten()
        unique = {op: op.build(**kwargs) for op in set(operators)}
        return [unique[op] for op in operators]

    @classmethod
    def list_options(cls):
        """Return list of options"""
        return {
            "max_nstate": "(int) maximum number of phase states",
            "squeeze": "(bool) activate operators merging",
            "prune": "(bool or list of variables) activate partials pruning",
            "prune_threshold": "(float) parameter for partials pruning",
        }

    def parse_string_operator(self, op):
        """Check and replace string operator with VirtualOperator objects"""
        if isinstance(op, str):
            if op == "ADC":
                return operators.ADC
            elif op == "SPOILER":
                return operators.SPOILER
            elif op == "RESET":
                return operators.RESET
        raise TypeError(f"Invalid string operator: {op}")

    def parse_options(self, **kwargs):
        """parse simulate options"""
        opts = {**self.options, **kwargs}
        if "prune" in opts:
            # setup partials pruning
            prune = opts.pop("prune", False)
            threshold = opts.pop("prune_threshold", 1e-5)
            if prune:
                variables = None if prune == True else prune
                pruner = diff.PartialsPruner(variables=variables, threshold=threshold)
                opts["callback"] = pruner
        return opts

    def adc_times(self, **kwargs):
        """Return sequence adc times (cf. operator's `duration` keyword)"""
        ops = self.build(**kwargs)
        return functions.get_adc_times(ops)

    def simulate(self, *, probe=None, gradient=None, hessian=None, **kwargs):
        """Simulate sequence

        Args:
            gradient: list of variables for which to compute 1st order derivatives
            hessian: list of variable pairs for which to compute 2nd order derivatives
            probe: alternative probing function/StateMatrix's attribute
            **kwargs: variables' values
                and additional options, cf. `sequence.list_options` method
        """
        # get argument values
        arguments = {var: kwargs.pop(var) for var in self.variables}
        # build operators
        ops = self.build(gradient=gradient, hessian=hessian, **arguments)
        # simulate
        options = self.parse_options(**kwargs)
        return functions.simulate(ops, probe=probe, **options)

    def derive(self, variable, *, probe="F0", **kwargs):
        """Simulate sequence derivation"""
        if not probe in base.Probe.SM_LOCALS:
            raise ValueError(f"Invalid probe: {probe}")

        if variable == "magnitude":
            _probe = lambda sm: getattr(sm, probe)
        else:
            _probe = lambda sm: getattr(sm.gradient[variable], probe)
        return self.simulate(probe=_probe, gradient=variable, **kwargs)

    def signal(self, **kwargs):
        """Signal simulation

        Args:
            **kwargs: variables' values and simulate options
                (cf. Sequence.list_options())

        Returns:
            signal: (... x nADC) signal ndarray
        """
        probe = base.Probe("F0")
        signal = self.simulate(probe=probe, **kwargs)
        return np.moveaxis(signal, 0, -1)  # move adc axis to last position

    def gradient(self, variables=None, **kwargs):
        """Signal's gradient/Jacobian simulation

        Args:
            variables: list of Jacobian's column variables
            **kwargs: variables' values and simulate options
                (cf. Sequence.list_options())

        Returns:
            signal: (... x nADC) signal ndarray
            jac: (... x nADC x nVars) Jacobian ndarray
        """
        if not variables:
            variables = list(self.variables)
        elif isinstance(variables, str):
            variables = [variables]
        probe = [
            base.Probe("F0"),
            diff.Jacobian(variables),
        ]
        signal, gradient = self.simulate(probe=probe, gradient=variables, **kwargs)
        # move adc/gradient axes to last position
        return np.moveaxis(signal, 0, -1), np.moveaxis(gradient, [0, 1], [-2, -1])

    def hessian(self, variables1=None, variables2=None, **kwargs):
        """Simulate signal's Hessian matrix

        Args:
            variables1: list of variables in the Hessian matrix
            variables2: if provided, columns of the Hessian matrix.
                If not provided, `variables1` is used for both rows and columns.
            **kwargs: variables' values and simulate options
                (cf. Sequence.list_options())

        Returns:
            signal: (... x nADC) signal ndarray
            jac: (... x nADC x nVars) Jacobian ndarray
            hes: (... x nADC x nVars1 x nVars2) Hessian ndarray
        """
        if not variables1:
            variables1 = list(self.variables)
        elif isinstance(variables1, str):
            variables1 = [variables1]
        if variables2 and isinstance(variables2, str):
            variables2 = [variables2]

        # output values
        probe = [
            base.Probe("F0"),
            diff.Jacobian(variables1),
            diff.Hessian(variables1, variables2),
        ]
        # simulate
        pairs = [(v1, v2) for v2 in (variables2 or variables1) for v1 in variables1]
        sig, grad, hess = self.simulate(probe=probe, hessian=pairs, **kwargs)
        # move adc/gradient axes to last position
        return (
            np.moveaxis(sig, 0, -1),
            np.moveaxis(grad, [0, 1], [-2, -1]),
            np.moveaxis(hess, [0, 1, 2], [-3, -2, -1]),
        )

    def crlb(
        self,
        variables=None,
        gradient=None,
        *,
        weights=None,
        log=False,
        **kwargs,
    ):
        """Cramer-Rao lower bound for specified variables

        Args:
            variables: list of variables used in CRLB calculation
            gradient: if provided, columns of the CRLB's Jacobian
            weights: CRLB weights (same length as `variables`)
            log: True/[False]: returns log10(CRLB)
            **kwargs: variables' values and simulate options
                (cf. Sequence.list_options())

        Returns:
            crlb: CRLB scalar (or ndarray if n-dimensional variable values passed)
        """
        if not gradient:
            _, J = self.gradient(variables, **kwargs)
            H = None
        else:
            variables2 = variables if gradient == True else gradient
            _, J, H = self.hessian(variables, variables2, **kwargs)
        return optim.crlb(J, H=H, W=weights, log=log)

    def confint(self, sse, variables=None, *, alpha=0.05, **kwargs):
        """return 95% confidence interval given sum of squared error"""
        # compute jacobian
        signal, J = self.gradient(variables, **kwargs)

        # check dimensions
        sse = np.asarray(sse)
        if signal.shape[:-1] not in (sse.shape, (1,)):
            raise ValueError(f"Invalid shape of `sse`: {sse.shape}")

        return optim.confint(sse, J, alpha=alpha)


#
# Virtual operators


class VirtualOperator(abc.ABC):
    """Sequence building block"""

    PARAMETERS = []
    DEFAULTS = {}
    OPERATOR = base.Operator
    # flags
    DIFFERENTIABLE = False

    @abc.abstractmethod
    def _build(self, **kwargs):
        """build true (non-virtual) operator"""

    def __init__(self, *args, **kwargs):
        kwargs = {**self.DEFAULTS, **kwargs}
        positionals = dict(zip(self.PARAMETERS, args))
        keywords = {
            name: kwargs.pop(name) for name in set(self.PARAMETERS) - set(positionals)
        }
        parameters = {**positionals, **keywords}
        self._expressions = {
            name: as_expression(parameters[name]) for name in parameters
        }
        self._options = {
            name: kwargs[name] for name in set(kwargs) - set(self.PARAMETERS)
        }
        self.variables = {
            var for expr in self._expressions.values() for var in expr.variables
        }

    def __hash__(self):
        return hash(
            (
                tuple(self._expressions),
                tuple(self._expressions.values()),
                tuple(self._options),
                tuple(self._options.values()),
            )
        )

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __call__(self, **kwargs):
        """short for 'fix'"""
        return self.fix(**kwargs)

    def flatten(self):
        """return self as list of virtual operators"""
        return [self]

    def fix(self, **kwargs):
        """fix some of the variables"""
        # update expressions and options with new kwargs
        kwargs = {**self._expressions, **self._options, **kwargs}
        return type(self)(**kwargs)

    def build(self, *, gradient=None, hessian=None, **kwargs):
        """return (non-virtual) EPG operator"""
        missing = self.variables - set(kwargs)
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        # parse values
        args = dict(self._options)
        args.update({name: expr(**kwargs) for name, expr in self._expressions.items()})

        # non differenciable operator
        if not self.DIFFERENTIABLE:
            return self._build(**args)

        # else: differentiable operator

        if hessian:
            if isinstance(hessian, str):
                hessian = [hessian]
            elif isinstance(hessian, list):
                hessian = flatten_list(hessian)
            if all(isinstance(item, str) for item in hessian):
                hessian = {pair: {} for pair in diff.combinations(hessian)}
            elif all(tuple(map(type, item)) == (str, str) for item in hessian):
                hessian = {pair: {} for pair in hessian}
            else:
                raise TypeError(f"Invalid hessian: {hessian}")
            # filter hessian
            hessian = {
                pair: hessian[pair] for pair in hessian if set(pair) & self.variables
            }
            if not gradient:
                # fill gradient if not set
                unique = set()
                gradient = [
                    var
                    for pair in hessian
                    for var in pair
                    if not (var in unique or unique.add(var))
                ]

        if gradient:
            if isinstance(gradient, str):
                gradient = [gradient]
            elif not all(isinstance(item, str) for item in gradient):
                raise TypeError(f"Invalid gradient: {gradient}")
            gradient = {
                variable: {
                    param: expr.derive(variable, **kwargs)
                    for param, expr in self._expressions.items()
                    if variable in expr.variables
                }
                for variable in gradient
            }

        if gradient or hessian:
            args["gradient"] = gradient
            args["hessian"] = hessian
        return self._build(**args)

    def __repr__(self):
        opname = type(self).__name__
        positionals = [p for p in self.PARAMETERS if not p in self.DEFAULTS]
        keywords = [p for p in self.PARAMETERS if p in self.DEFAULTS]
        exprs = self._expressions
        args = ", ".join(str(exprs[name]) for name in positionals)
        if keywords:
            args += ", " + ", ".join(f"{name}={exprs[name]}" for name in keywords)
        return f"{opname}({args})"


#
# Expressions


def document(ref):
    def wrapper(cls):
        cls.__doc__ = ref.__doc__
        if not "__init__" in vars(cls):

            def init(self, *args, **kwargs):
                super(cls, self).__init__(*args, **kwargs)

            cls.__init__ = init
        cls.__init__.__doc__ = ref.__init__.__doc__
        return cls

    return wrapper


class operators(types.SimpleNamespace):
    """namespace for virtual operators"""

    @document(base.T)
    class T(VirtualOperator):
        PARAMETERS = ["alpha", "phi"]
        DIFFERENTIABLE = True

        def _build(self, alpha, phi, **kwargs):
            return diff.T(alpha, phi, **kwargs)

    @document(base.E)
    class E(VirtualOperator):
        PARAMETERS = ["tau", "T1", "T2", "g"]
        DEFAULTS = {"g": 0}
        DIFFERENTIABLE = True

        def _build(self, tau, T1, T2, g, **kwargs):
            return diff.E(tau, T1, T2, g, **kwargs)

    @document(base.S)
    class S(VirtualOperator):
        PARAMETERS = ["k"]
        DIFFERENTIABLE = True

        def _build(self, k, **kwargs):
            return diff.S(k, **kwargs)

    # utilities

    @document(base.Probe)
    class Probe(VirtualOperator):
        DEFAULTS = {"obj": "F0"}

        def __repr__(self):
            probe = self._options["obj"]
            return self._options.get("name", f"Probe({probe})")

        def _build(self, **kwargs):
            return base.Probe(**kwargs)

    @document(base.Adc)
    class Adc(VirtualOperator):
        PARAMETERS = ["phase", "weights"]

        def __init__(self, attr="F0", *, phase=None, weights=None, **kwargs):
            super().__init__(attr=attr, phase=phase, weights=weights, **kwargs)

        def _build(self, **kwargs):
            return base.Adc(**kwargs)

        def __repr__(self):
            attr = self._options["attr"]
            phase = self._expressions.get("phase")
            return self._options.get("name", f"Probe({attr}, phase={phase})")

    @document(base.Spoiler)
    class Spoiler(VirtualOperator):
        def _build(self, **kwargs):
            return base.Spoiler(**kwargs)

    @document(base.Reset)
    class Reset(VirtualOperator):
        def _build(self, **kwargs):
            return base.Reset(**kwargs)

    @document(base.Wait)
    class Wait(VirtualOperator):
        PARAMETERS = ["duration"]

        def _build(self, **kwargs):
            return base.Wait(**kwargs)

    # other operators

    class RFPulse(VirtualOperator):
        PARAMETERS = ["values", "duration", "rf", "alpha", "phi", "T1", "T2", "g"]
        DEFAULTS = {
            "rf": None,
            "alpha": None,
            "phi": None,
            "T1": None,
            "T2": None,
            "g": None,
        }
        DIFFERENTIABLE = False  # for now

        def _build(self, values, duration, **kwargs):
            """init rf-pulse operator"""
            return rfpulse.RFPulse(values, duration, **kwargs)


#
# Expression: minimal formal calculus

"""TODO
- remove helpers from methods definitions
- a Function must have a fixed number of variables
- partial derivatives must unambiguously associated with the variables
- remove lambda in maths

"""


def as_function(func=None, *, repr=None, derivative=None):
    """return Function object"""

    def decorator(func):
        if isinstance(func, Function):
            return func
        else:
            return Function(func, repr=repr, derivative=derivative)

    if func is not None:
        return decorator(func)
    return decorator


def as_expression(obj):
    """return Expression/Variable/Constant"""
    if isinstance(obj, Expression):
        return obj
    elif obj is None:
        return NoneExpr()
    elif isinstance(obj, str):
        return Variable(obj)
    else:
        return Constant(obj)


class Function:
    """Function class"""

    def __init__(self, func, repr=None, derivative=None, **kwargs):
        """initialize function"""
        self.func = func
        self.repr = repr
        self.kwargs = {} if not kwargs else kwargs
        self.partials = {}
        if derivative is not None:
            self.set_derivative(derivative)

    def __repr__(self):
        return self.represent()

    def __call__(self, *args, **kwargs):
        """helper to create an expression"""
        args = [as_expression(arg) for arg in args]
        return Expression(self, args, **kwargs)

    def represent(self, args=None):
        """represent function"""
        nest = isinstance(self.repr, str)
        args = [expr.represent(nest=nest) for expr in args] if args else []
        if self.repr is None:
            return f'{self.func.__name__}({", ".join(args)})'
        elif callable(self.repr):
            return self.repr(args)
        elif not args:
            args = ["·"] * 10
        return self.repr.format(*args)

    def execute(self, *args, **kwargs):
        """execute function"""
        return self.func(*args, **{**self.kwargs, **kwargs})

    def set_derivative(self, func, index=None):
        """set derivative function"""
        if isinstance(func, dict):
            [self.set_derivative(func[i], index=i) for i in func]
            return
        elif isinstance(func, list):
            [self.set_derivative(func[i], index=i) for i in range(len(func))]
            return

        func = as_function(func)
        if index is None:
            # single derivative function, no index
            self.partials[None] = func
        elif index is Ellipsis:
            # single derivative function, with index argument
            self.partials[Ellipsis] = func
        else:
            # dictionary of partial derivatives
            self.partials[index] = func

    def derive(self, index, *args, **kwargs):
        """return Expression of ith-derivative"""
        if not self.partials:
            raise RuntimeError(f"No derivative provided for function {self}")
        if None in self.partials:
            return self.partials[None](*args, **kwargs)
        elif Ellipsis in self.partials:
            return self.partials[Ellipsis](index, *args, **kwargs)
        elif index in self.partials:
            return self.partials[index](*args, **kwargs)
        else:
            raise RuntimeError(f"No derivative provided for argument #{index}")


class Expression:
    """Formal expression"""

    function = None
    arguments = ()

    def __init__(self, function, arguments, **kwargs):
        self.function = as_function(function)
        self.arguments = tuple(as_expression(a) for a in arguments)
        self.variables = {var for expr in self.arguments for var in expr.variables}

    def __repr__(self):
        return self.represent()

    def __eq__(self, other):
        return (self.function is other.function) & (self.arguments == other.arguments)

    def __hash__(self):
        return hash((self.function, self.arguments))

    def __call__(self, **kwargs):
        """solve expressions recursively"""
        missing = set(self.variables) - set(kwargs)
        if missing:
            raise ValueError(f"Missing values for variables: {missing}")
        args = [expression(**kwargs) for expression in self.arguments]
        return self.function.execute(*args)

    def represent(self, nest=False):
        repr = self.function.represent(self.arguments)
        nest = nest & isinstance(self.function.repr, str)
        return f"({repr})" if nest else repr

    # math

    def derive(self, variable, *, solve=True, **kwargs):
        """solve derive recursively"""
        derivative = None
        for i, expr in enumerate(self.arguments):
            if not variable in expr.variables:
                continue
            partial = expr.derive(variable, solve=False) * self.function.derive(
                i, *self.arguments
            )
            derivative = partial if derivative is None else derivative + partial
        if derivative is None:
            derivative = math.zeros()
        if not solve:
            return derivative
        # solve derivative
        return derivative(**kwargs)

    def __neg__(self):
        return math.neg(self)

    def __add__(self, other):
        return math.add(self, other)

    def __radd__(self, other):
        return math.add(other, self)

    def __sub__(self, other):
        return math.sub(self, other)

    def __rsub__(self, other):
        return math.sub(other, self)

    def __mul__(self, other):
        return math.mult(self, other)

    def __rmul__(self, other):
        return math.mult(other, self)

    def __truediv__(self, other):
        return math.div(self, other)

    def __rtruediv__(self, other):
        return math.div(other, self)

    def __pow__(self, other):
        return math.pow(self, other)

    def __rpow__(self, other):
        return math.pow(other, self)


class NoneExpr(Expression):
    variables = {}

    def __init__(self):
        pass

    def __call__(self, **kwargs):
        return None

    def derive(self, *args, **kwargs):
        return None

    def represent(self, nest=False):
        return "None"


class Constant(Expression):
    """Constant is an Expression building block"""

    def __init__(self, value, name=None):
        if np.isscalar(value):
            dtype = type(value)
        else:
            value = np.asarray(value)
            dtype = value.dtype
        if not np.issubdtype(dtype, np.number):
            raise TypeError(f"Invalid numeric value: '{value}'")
        self.value = value
        self.variables = set()
        self.name = name

    def __hash__(self):
        if np.isscalar(self.value):
            return hash(self.value)
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        return np.isclose(self.value, other.value)

    def represent(self, nest=False):
        nest = np.any(self.value < 0) & nest
        if self.name:
            repr = str(self.name)
        elif np.isscalar(self.value):
            repr = str(self.value)
        else:
            repr = f'array({"x".join(map(str, self.value.shape))})'
        return f"({repr})" if nest else repr

    def __call__(self, **kwargs):
        return self.value

    def derive(self, variable, *, solve=True, **kwargs):
        if not solve:
            return Constant(0 * self.value)
        return 0


class Variable(Expression):
    """Variable is an Expression building block"""

    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError(f"Variable's name must be name, not : {name}")
        self.name = name
        self.variables = {name}

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Variable) and other.name == self.name

    def represent(self, nest=False):
        return str(self.name)

    def __call__(self, **kwargs):
        try:
            value = kwargs[self.name]
        except KeyError:
            raise ValueError(f"Missing value for variable {self.name}")
        if not np.isscalar(value):
            return np.asarray(value)
        return value

    def derive(self, variable, *, solve=True, **kwargs):
        expr = Constant(1) if variable == self.name else Constant(0)
        if not solve:
            return expr
        return expr(**kwargs)


# class expr_math(types.SimpleNamespace):
class math(types.SimpleNamespace):
    @as_function(repr="0")
    def zeros(*args):
        shape = np.broadcast(*args).shape
        return 0 if not shape else np.zeros(shape)

    @as_function(repr="1", derivative=zeros)
    def ones(*args):
        shape = np.broadcast(*args).shape
        return 1 if not shape else np.ones(shape)

    @as_function(repr="-1", derivative=zeros)
    def minus_ones(*args):
        shape = np.broadcast(*args).shape
        return -1 if not shape else np.ones(shape)

    @as_function(repr="-{0}", derivative=minus_ones)
    def neg(a):
        return -a

    @as_function(repr="{0}", derivative=[ones, zeros])
    def first(a1, a2):
        return a1 + 0 * a2

    @as_function(repr="{1}", derivative=[zeros, ones])
    def second(a1, a2):
        return a2 + 0 * a1

    @as_function(repr="{0}+{1}", derivative=[ones, ones])
    def add(a1, a2):
        return a1 + a2

    @as_function(repr="{0}-{1}", derivative=[ones, minus_ones])
    def sub(a1, a2):
        return a1 - a2

    @as_function(repr="{0}*{1}", derivative=[second, first])
    def mult(a1, a2):
        return a1 * a2

    @as_function(repr="{0}/{1}")
    def div(a1, a2):
        return a1 / a2

    div.set_derivative(lambda a1, a2: 1 / a2 + 0 * a1, index=0)
    div.set_derivative(lambda a1, a2: -a1 / a2**2, index=1)

    @as_function(repr="{0}**{1}")
    def pow(a1, a2):
        return np.power(a1, a2)

    pow.set_derivative(lambda a1, a2: a2 * np.power(a1, a2 - 1), index=0)
    pow.set_derivative(lambda a1, a2: np.log(a1) * np.power(a1, a2), index=1)

    @as_function
    def log(a):
        return np.log(a)

    log.set_derivative(lambda a: 1 / a)

    @as_function
    def exp(arg):
        return np.exp(arg)

    exp.set_derivative(exp)

    @as_function
    def cos(arg):
        return np.cos(arg)

    @as_function
    def sin(arg):
        return np.sin(arg)

    sin.set_derivative(cos)
    cos.set_derivative(lambda a: -np.sin(a))

    @as_function
    def tan(arg):
        return np.tan(arg)

    tan.set_derivative(lambda a: 1 / np.cos(a) ** 2)

    @as_function
    def sqrt(arg):
        return np.sqrt(arg)

    sqrt.set_derivative(lambda a: 0.5 / np.sqrt(a))

    @as_function(derivative=zeros)
    def sign(arg):
        return np.sign(arg)

    @as_function(derivative=sign)
    def abs(arg):
        return np.abs(arg)


# misc
def flatten_list(items):
    """flatten list of items"""
    if not isinstance(items, list):
        return [items]

    flattened = []
    for item in items:
        flattened.extend(flatten_list(item))
    return flattened


# instanciate some utility operators
operators.ADC = operators.Adc(attr="F0", name="ADC")
operators.SPOILER = operators.Spoiler(name="Spoiler")
operators.RESET = operators.Reset(name="Reset")
