"""Sequence building tools

from epg.sequence import Variable, Sequence, operators

# Define variables
T1 = Variable("T1")
T2 = Variable("T2")
b1 = Variable("b1")

# Get (virtual) operators to use in sequence
T = operators.T
S = operators.S
E = operators.S

# Define a sequence as list of operators (with some variables in arguments)
ops = [T(90 * b1, 90), S(1), E(9.5, T1, T2), ...]
seq = Sequence(ops)

# Basic functions
seq.variables --> {"b1", "T1", "T2"} # sequence's variables

# Simulate signal, Jacobian and Hessian matrix (tensor)
signal = seq.signal(b1=0.9, T1=1000, T2=30)
# equivalently: seq(b1=0.9, T1=1000, T2=30)
signal, jacobian = seq.jacobian(['T2', 'b1'], b1=0.9, T1=1000, T2=30)
sig, grad, hess = seq.hessian(['T2', 'b1'], b1=0.9, T1=1000, T2=30)

# Other functions

# Only build (non-virtual) EPG operators, without simulation
seq.build({b1: 0.9, T1: 1000, T2: 30})
    --> [epg.T(0.9*90, 90), epg.S(1), epg.E(0.5, 1000, 30), ...]

# Run epg.functions.simulate (more flexible than seq.signal)
seq.simulate({b1: 0.9, T1: 1000, T2: 30}, probe='Z0') --> value

# CRLB (sequence optimization objective function)
seq.crlb(['T2', 'b1'])(b1=0.9, T1=1000, T2=30)

# Confidence intervals
seq.confint(obs, ['T2', 'b1'])(b1=0.9, T1=1000, T2=30)


# Tips

# Operator's variables can be passed directly as string
seq = Sequence([E('tau', T1, T2), T('alpha', phi), ...])

# ADC, SPOILER, RESET operators can be passed as string
`Sequence([op1, op2, ..., 'ADC', 'SPOILER'])

# Avoid computing unnecessary partial derivatives in the Hessian tensor
# by passing different variables in rows (axis=-2) and columns (axis=-1)
seq.hessian([var1, var2], [var3])(...)

# Compute the gradient of the crlb w/r to given variable(s)
seq.crlb([var1, var2], gradient=[var3])(...)

# keyword options to epg.functions.simulate can be set at different places:
seq = Sequence(ops, options={'max_nstate': 10, 'disp': True})
seq.signal(options={'max_nstate': 10, 'disp': True})(...)

# Calling `signal`, `jacobian`, `hessian`, `crlb` or `confint`
# without passing the variable's values returns a function
# with arguments: (values_dict=None, **values). For instance:
seq.crlb(variables1, gradient=variables2, **options)({var1: value1}, var2=value2)


"""

import abc
import types
import numpy as np
from . import operators as epgops, functions as epgfuncs, stats


class Sequence:
    """Sequence building object"""

    def __init__(self, ops=[], *, name=None, options=None):
        """Build sequence from a list of virtual operators

        Args:
            ops: list of virtual operators
            name: sequence's name
        """
        ops = flatten(ops)
        ops = self.check(ops)
        self.operators = ops  # operator list
        self.name = name  # display name
        self.options = options or {}  # simulate options

    def __len__(self):
        return len(self.operators)

    def __iter__(self):
        return iter(self.operators)

    def __getitem__(self, item):
        return self.operators[item]

    def __setitem__(self, item, op):
        if isinstance(op, Sequence):
            ops = op.operators
        elif isinstance(op, list):
            ops = self.check(op)
        else:  # single insert
            ops = self.check([op])
            item = slice(item, item + 1)
        # assume bulk insert
        self.operators[item] = ops

    def __delitem__(self, item):
        del self.operators[item]

    def __add__(self, other):
        if not isinstance(other, Sequence):
            raise ValueError(f"Expecting Sequence, not: {type(other)}")
        return self.copy(self.operators + other.operators)

    def __repr__(self):
        return self.name if self.name else f"Sequence({len(self)})"

    def __call__(self, *args, **kwargs):
        """call signal"""
        return self.signal(*args, **kwargs)

    @property
    def variables(self):
        """set of variables in the sequence"""
        return {var for op in self.operators for var in op.variables}

    def check(self, ops):
        """Check and convert provided operators"""
        # replace string objects with virtual operators
        ops = [operators.STR_OPS.get(op, op) for op in ops]
        # check operator type
        invalid = {op for op in ops if not isinstance(op, VirtualOperator)}
        if invalid:
            raise ValueError(f"Invalid operator(s): {invalid}")
        return ops

    def copy(self, ops=None, **kwargs):
        """Copy sequence"""
        ops = ops or self.operators
        name = kwargs.get("name", self.name)
        return Sequence(ops, name=name)

    def build(self, values=None, *, order1=None, order2=None):
        """build EPG operators"""
        # check jacobian and hessian
        variables = self.variables
        if order1:
            order1 = [var for var in order1 if var != "magnitude"]
            invalid = set(order1) - variables
            if invalid:
                raise ValueError(f"Unknown variable(s) in order1: {invalid}")
        if order2:
            order2 = [pair for pair in order2 if not "magnitude" in pair]
            hessvars = {var for pair in order2 for var in pair}
            invalid = hessvars - variables
            if invalid:
                raise ValueError(f"Unknown variable(s) in order2: {invalid}")
            if not order1:
                order1 = list(hessvars)

        # build operators
        unique = {}  # unique operators
        return [
            unique.setdefault(op, op.build(values or {}, order1=order1, order2=order2))
            for op in self.operators
        ]

    def simulate(self, values=None, *, order1=None, order2=None, probe=None, **kwargs):
        """Run epg.functions.simulate() on sequence's operators.

        Args:
            values: dict of variable's values
            kwargs: cf. epg.functions.simulate
        """
        options = {**self.options, **kwargs}
        ops = self.build(values, order1=order1, order2=order2)
        return epgfuncs.simulate(ops, probe=probe, **options)

    def adc_times(self, **values):
        """Return adc times"""
        ops = self.build(values=values)
        return epgfuncs.get_adc_times(ops)

    def signal(self, *, options={}, **values):
        """Simulate the sequence's signal

        Args:
            **values: variables' values

        Returns:
            signal: (... x nADC) signal ndarray
        """
        probe = epgops.Probe("F0")

        def signal(valuesdict=None, **values):
            values.update(valuesdict or {})
            sim = self.simulate(values, probe=probe, asarray=True, **options)
            return np.moveaxis(sim, 0, -1)

        return signal(**values) if values else signal

    def jacobian(self, variables, *, options={}, **values):
        """Simulate the signal's Jacobian matrix (tensor)

        Args:
            variables: list of variables of the Jacobian matrix's partials
            **kwargs: variables' values and simulate options

        Returns:
            signal: (... x nADC) signal ndarray
            jac: (... x nADC x nVars) Jacobian ndarray
        """
        if isinstance(variables, str):
            variables = [variables]
        probe = [epgops.ADC, epgops.Jacobian(list(variables))]

        def jacobian(valuesdict=None, **values):
            values.update(valuesdict or {})
            sim, jac = self.simulate(
                values, order1=variables, probe=probe, asarray=True, **options
            )
            return np.moveaxis(sim, 0, -1), np.moveaxis(jac, [0, 1], [-2, -1])

        return jacobian(**values) if values else jacobian

    def hessian(self, variables1, variables2=None, *, options={}, **values):
        """Simulate the signal's Hessian matrix (tensor)

        Args:
            variables1: list of variables of the Hessian matrix's partials
            variables2: if provided, variables of the Hessian matrix's 2nd partials.
                If not provided, variables is used for both 1st and 2nd partials.
            **kwargs: variables' values and simulate options

        Returns:
            signal: (... x nADC) signal ndarray
            jac: (... x nADC x nVars1) Jacobian ndarray
            hes: (... x nADC x nVars1 x nVars2) Hessian ndarray
        """
        if isinstance(variables1, str):
            variables1 = [variables1]
        if variables2 is None:
            variables2 = variables1
        elif isinstance(variables2, str):
            variables2 = [variables2]

        probe = [
            epgops.ADC,
            epgops.Jacobian(list(variables1)),
            epgops.Hessian(list(variables1), list(variables2)),
        ]
        pairs = [(v1, v2) for v1 in variables1 for v2 in variables2 if v1 <= v2]

        def hessian(valuesdict=None, **values):
            values.update(valuesdict or {})
            sim, jac, hess = self.simulate(
                values,
                order1=variables1,
                order2=pairs,
                probe=probe,
                asarray=True,
                **options,
            )
            return (
                np.moveaxis(sim, 0, -1),
                np.moveaxis(jac, [0, 1], [-2, -1]),
                np.moveaxis(hess, [0, 1, 2], [-3, -2, -1]),
            )

        return hessian(**values) if values else hessian

    def crlb(
        self, variables, *, gradient=None, weights=None, log=False, sigma2=1, options={}
    ):
        """Cramer-Rao lower bound for given variables

        Args:
            variables: list of variables used in CRLB calculation
            gradient: if provided, variables for the CRLB's gradient
            weights: CRLB weights (same length as `variables`)
            log: True/[False]: returns log10(CRLB)

        Returns:
            CRLB function with values passed as keywords or dictionary,
            with return values:
                crlb: CRLB scalar (or ndarray if n-dimensional variable values passed)
            if `gradient` is provided:
                crlb, Jcrlb: where Jcrlb is the gradient of the crlb w/r gradient variables.
        """

        def crlb(valuesdict=None, **values):
            values.update(valuesdict or {})
            hess = None
            if not gradient:
                _, jac = self.jacobian(variables, options=options)(values)
            else:
                variables2 = variables if gradient is True else list(gradient)
                _, jac, hess = self.hessian(variables, variables2, options=options)(
                    values
                )
            return stats.crlb(jac, H=hess, W=weights, log=log, sigma2=sigma2)

        return crlb

    def confint(self, obs, variables, *, conflevel=0.95, return_cband=False):
        """return 95% confidence interval for given observation

        Args:
            obs: observations (... x nADC) ndarray
            variables: (nVar) list of variables for the confidence intervals

        Returns:
            confint function with values passed as keywords or dictionary,
            with return values:
                cints: 1/2 width of confidence intervals (... x nVar) ndarray
                if return_cband == True,
                cband: confidence band of prediction (... x nADC) ndarray
        """
        obs = np.asarray(obs)

        def confint(valuesdict=None, **values):
            values.update(valuesdict or {})
            # compute prediction and jacobian
            pred, jac = self.jacobian(variables, **values)
            # check dimensions
            if obs.shape != pred.shape:
                raise ValueError(f"Mismatch between observation and prediction shapes")
            # compute confidence intervals
            cints, cband = stats.confint(obs, pred, jac, conflevel=conflevel)
            if return_cband:
                return cints, cband
            return cints

        return confint


class VirtualOperator(abc.ABC):
    """Virtual operator base class"""

    OPERATOR = None
    POSITIONALS = []
    KEYWORDS = []
    OPTIONS = []

    @property
    def variables(self):
        """set of variables"""
        variables = set()
        for expr in self.positionals + list(self.keywords.values()):
            variables |= {var for var in expr.variables}
        return variables

    def __init__(self, *args, **kwargs):
        # list positionals
        positionals = list(args) + [
            kwargs.pop(key) for key in set(kwargs) & set(self.POSITIONALS)
        ]
        keywords = {key: kwargs.pop(key) for key in set(kwargs) & set(self.KEYWORDS)}
        options = kwargs
        # check options
        if not Ellipsis in self.OPTIONS:
            unknown = set(options) - set(self.OPTIONS)
            if unknown:
                raise ValueError(f"Unknown option(s): {options}")
        # make expressions
        for i in range(len(positionals)):
            positionals[i] = to_expression(positionals[i])
        for key in keywords:
            keywords[key] = to_expression(keywords[key])
        self.positionals = positionals
        self.keywords = keywords
        self.options = options

    def __getattr__(self, attr):
        try:
            idx = self.POSITIONALS.index(attr)
            return self.positionals[idx]
        except ValueError:
            pass
        if attr in self.keywords:
            return self.keywords[attr]
        elif attr in self.options:
            return self.options[attr]
        return getattr(super(), attr)

    def __call__(self, /, **values):
        return self.fix(values)

    def fix(self, values):
        """fix some values"""
        args = [arg.fix(**values) for arg in self.positionals]
        keywords = {key: value.fix(**values) for key, value in self.keywords.items()}
        kwargs = {**keywords, **self.options}
        return type(self)(*args, **kwargs)

    def build(self, values={}, *, order1=None, order2=None):
        """build (non-virtual) EPG operator"""
        # solve expressions
        args = [arg(**values) for arg in self.positionals]
        keywords = {key: value(**values) for key, value in self.keywords.items()}
        kwargs = {**keywords, **self.options}

        # build operator
        if not (order1 or order2):
            return self.OPERATOR(*args, **kwargs)

        # build order1 and order2 dicts
        order1 = set(order1 or [])
        order2 = {tuple(sorted(pair)) for pair in (order2 or [])}
        hesvars = {var for pair in order2 for var in pair}

        exprs = list(zip(self.POSITIONALS, self.positionals))
        exprs += [(name, self.keywords[name]) for name in self.KEYWORDS]
        _order1, _order2 = {}, {}
        for param, expr in exprs:
            # get argument's variables
            variables = set(map(str, expr.variables))
            for var in variables & (order1 | hesvars):
                # 1st order derivatives
                d1param = expr.derive(var, **values)
                _order1.setdefault(var, {}).update({param: d1param})
            for pair in order2:
                if pair[0] in variables and pair[1] in variables:
                    # 2nd order derivative
                    _order2.setdefault(pair, {})
                    d2param = expr.derive(pair[0]).derive(pair[1], **values)
                    if not np.allclose(d2param, 0):
                        _order2[pair].update({param: d2param})
                elif pair[0] in variables or pair[1] in variables:
                    # 1st order cross derivatives
                    _order2.setdefault(pair, {})

        if _order1:
            kwargs["order1"] = _order1
        if _order2:
            kwargs["order2"] = _order2
        return self.OPERATOR(*args, **kwargs)

    def __repr__(self):
        args = ", ".join(repr(arg) for arg in self.positionals)
        return f"{self.OPERATOR.__name__}({args})"


class VirtualOperatorInstance(VirtualOperator):
    """wrapper for ADC, SPOILER, etc."""

    OPERATOR = None
    VARIABLES = []
    KEYWORDS = []
    OPTIONS = None

    def __init__(self, operator, positionals=None, options=None):
        self.operator = operator
        self.positionals = [getattr(operator, attr) for attr in positionals] or []
        self.keywords = {}
        self.options = {opt: getattr(operator, opt) for opt in options}

    def fix(self, *args):
        raise NotImplementedError(f"Nothing to fix in {self}")

    def build(self, *args, **kwargs):
        return self.operator

    def __repr__(self):
        return repr(self.operator)


def virtual_operator(op, pos=[], kw=[], opt=[]):
    """make virtual operator"""
    if isinstance(op, epgops.Operator):
        return VirtualOperatorInstance(op, positionals=pos, options=opt)

    class Op(VirtualOperator):
        OPERATOR = op
        POSITIONALS = pos
        KEYWORDS = kw
        OPTIONS = opt

        def __init__(self, *args, **kwargs):
            """place holder for docstring"""
            super().__init__(*args, **kwargs)

    Op.__name__ = op.__name__
    Op.__doc__ = op.__doc__
    Op.__init__.__doc__ = op.__init__.__doc__
    return Op


class operators(types.SimpleNamespace):
    """availabel virtual operators"""

    _std = ["name", "duration"]
    _diff = ["order1", "order2"]

    E = virtual_operator(epgops.E, ["tau", "T1", "T2", "g"], [], _diff + _std)
    P = virtual_operator(epgops.P, ["g"], [], _diff + _std)
    R = virtual_operator(epgops.P, ["rT", "rL", "r0"], [], _diff + _std)
    T = virtual_operator(epgops.T, ["alpha", "phi"], [], _diff + _std)
    Phi = virtual_operator(epgops.Phi, ["phi"], [], _diff + _std)
    S = virtual_operator(epgops.S, ["k"], [], _std)
    D = virtual_operator(epgops.D, ["tau", "D", "k"], [], _std)
    X = virtual_operator(epgops.X, ["tau", "khi"], ["T1", "T2", "g"], _std)

    # utilities
    Adc = virtual_operator(
        epgops.Adc, [], ["phase"], ["attr", "reduce", "weights"] + _std
    )
    Wait = virtual_operator(epgops.Wait, ["duration"], [], ["name"])
    Offset = virtual_operator(epgops.Offset, ["duration"], [], ["name"])
    Spoiler = virtual_operator(epgops.Spoiler, [], [], _std)
    Reset = virtual_operator(epgops.Reset, [], [], _std)
    System = virtual_operator(epgops.System, [], [], _std + [None])

    # default operators
    ADC = virtual_operator(epgops.ADC, [], [], _std)
    NULL = virtual_operator(epgops.NULL, [], [], _std)
    SPOILER = virtual_operator(epgops.SPOILER, [], [], _std)
    RESET = virtual_operator(epgops.RESET, [], [], _std)

    # string operators
    STR_OPS = {
        "ADC": ADC,
        "NULL": NULL,
        "SPOILER": SPOILER,
        "RESET": RESET,
    }


#
# order1/order2 keywords
def parse_order12(seq, order1=None, order2=None):
    """parse order1/order2 keywords"""


#
# Expressions


def to_expression(obj):
    """return Expression"""
    if isinstance(obj, Expression):
        return obj
    elif isinstance(obj, str):
        return Variable(obj)
    else:
        return Constant(obj)


class Expression:
    """Mathematical expression"""

    def __init__(self, function, arguments):
        self.function = function
        self.arguments = list(arguments)

    def __repr__(self):
        args = [repr(arg) for arg in self.arguments]
        return self.function.repr(args)

    def __call__(self, /, **values):
        """compute expression"""
        # solve arguments
        values = [arg(**values) for arg in self.arguments]
        # execute function
        return self.function.execute(*values)

    def fix(self, /, **values):
        """transform variables into constants"""
        arguments = [arg.fix(**values) for arg in self.arguments]
        return Expression(self.function, arguments)

    def map(self, mapping):
        """map expression variables"""
        if not mapping or not self.arguments:
            return self
        mapping = {str(key): value for key, value in mapping.items()}
        args = []
        for arg in self.arguments:
            if isinstance(arg, Variable):
                args.append(mapping.get(str(arg), arg))
            elif arg.arguments:
                args.append(arg.map(mapping))
            else:
                args.append(arg)
        return Expression(self.function, args)

    def derive(self, variable, /, **kwargs):
        """compute derivative expression"""
        variable = str(variable)
        d_expr = Constant(0)
        for i, arg in enumerate(self.arguments):
            if variable in map(str, arg.variables):
                # derive function
                partial = self.function.derive(i)
                # solve proxy variables
                mapping = dict(zip(partial.proxies, self.arguments))
                partial = partial.map(mapping)
                # derive argument
                if not isinstance(arg, Variable):
                    partial = arg.derive(variable) * partial
                # add partial derivatives
                if d_expr is None:
                    d_expr = partial
                else:
                    d_expr += partial
        return d_expr(**kwargs) if kwargs else d_expr

    @property
    def variables(self):
        unique = {var.name: var for arg in self.arguments for var in arg.variables}
        return set(unique.values())

    @property
    def proxies(self):
        return sorted(
            {var for var in self.variables if isinstance(var, Proxy)},
            key=lambda var: var.position,
        )

    # standard operators
    def __neg__(self):
        return Expression(functions.neg, [self])

    def __abs__(self):
        return Expression(functions.abs, [self])

    def __add__(self, other):
        other = to_expression(other)
        return Expression(functions.add, [self, other])

    def __radd__(self, other):
        other = to_expression(other)
        return Expression(functions.add, [other, self])

    def __sub__(self, other):
        other = to_expression(other)
        return Expression(functions.sub, [self, other])

    def __rsub__(self, other):
        other = to_expression(other)
        return Expression(functions.sub, [other, self])

    def __mul__(self, other):
        other = to_expression(other)
        return Expression(functions.mul, [self, other])

    def __rmul__(self, other):
        other = to_expression(other)
        return Expression(functions.mul, [other, self])

    def __truediv__(self, other):
        other = to_expression(other)
        return Expression(functions.div, [self, other])

    def __rtruediv__(self, other):
        other = to_expression(other)
        return Expression(functions.div, [other, self])

    def __pow__(self, other):
        other = to_expression(other)
        return Expression(functions.pow, [self, other])

    def __rpow__(self, other):
        other = to_expression(other)
        return Expression(functions.pow, [other, self])


class Constant(Expression):
    """A scalar constant"""

    function = None
    arguments = []
    variables = set()

    def __init__(self, value, name=None):
        if isinstance(value, (np.ndarray, list)):
            value = np.asarray(value)
            name = name or f'arr[{", ".join(map(str, value.shape))}]'
        self.value = value
        self.name = name or f"{value}"

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        other = other.value if isinstance(other, Constant) else other
        return np.all(self.value == other)

    def __hash__(self):
        return hash(self.value)

    def __call__(self, /, **kwargs):
        return self.value

    def fix(self, /, **kwargs):
        return self

    def derive(self, variable, /, **kwargs):
        expr = Constant(0.0)
        return expr(**kwargs) if kwargs else expr


class Variable(Expression):
    """A scalar variable"""

    function = None
    arguments = []
    variables = set()

    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError(f"Expecting str, not {type(name)}")
        self.name = name
        self.variables = {self}

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        other = other.name if isinstance(other, Variable) else other
        return self.name == other

    def __hash__(self):
        return hash(self.name)

    def __call__(self, /, **kwargs):
        if not self.name in kwargs:
            raise ValueError(f"Missing variable: {self.name}")
        value = kwargs[self.name]
        if isinstance(value, (np.ndarray, list)):
            return np.asarray(value)
        return value

    def fix(self, /, **kwargs):
        if self.name in kwargs:
            return Constant(kwargs[self.name])
        return self

    def derive(self, variable, /, **kwargs):
        expr = Constant(1.0) if variable == self.name else Constant(0.0)
        return expr(**kwargs) if kwargs else expr


class Proxy(Variable):
    """proxy scalar argument"""

    def __init__(self, position):
        if not isinstance(position, int):
            raise ValueError(f"Expecting int, not {type(position)}")
        self.position = position
        self.name = f"<arg{position}>"
        self.variables = {self}

    def __call__(self, /, **kwargs):
        raise NotImplementedError(f"Cannot solve a proxy variable")

    def derive(self, variable, /, **kwargs):
        raise NotImplementedError(f"Cannot derive a proxy variable")


class Function:
    """Function wrapper including derivatives"""

    def __init__(self, function, *, derivatives=None, name=None, fmt=None, kwargs=None):
        if not callable(function):
            raise ValueError(f"Expecting callable, not {type(function)}")
        self.function = function
        self.kwargs = kwargs or {}
        self.name = name or function.__name__
        self.fmt = fmt or "{name}({args})"

        if derivatives:
            if not isinstance(derivatives, list):
                raise ValueError(f"Expecting list of derivatives, not {type(function)}")
            elif not all(
                isinstance(func, (type(None), Expression)) for func in derivatives
            ):
                types = [type(func) for func in derivatives]
                raise ValueError(f"Expecting list of None or Function, not {types}")
        self.derivatives = derivatives

    def repr(self, args):
        strargs = {"args": ", ".join(args)}
        strargs.update({f"arg{i + 1}": arg for i, arg in enumerate(args)})
        return self.fmt.format(name=self.name, **strargs)

    def __repr__(self):
        return self.name

    def execute(self, *args):
        """execute the function"""
        return self.function(*args, **self.kwargs)

    def __call__(self, *args):
        """return expression"""
        args = [to_expression(arg) for arg in args]
        return Expression(self, args)

    def derive(self, i):
        """return ith derivative expression"""
        if not self.derivatives:
            raise ValueError(f"Undefined derivatives")
        expr = self.derivatives[i]
        if not expr:
            raise ValueError(f"Undefined {i}-th derivative")
        return expr


# helpers
def tofunction(derivatives=None, fmt=None):
    def wrapper(func):
        f = Function(func, derivatives=derivatives, fmt=fmt)
        return f

    return wrapper


class functions(types.SimpleNamespace):
    """available functions"""

    p1, p2 = Proxy(1), Proxy(2)

    # left and right
    @tofunction([Constant(1), Constant(0)], fmt="{arg1}")
    def left(v1, v2):
        return v1

    @tofunction([Constant(0), Constant(1)], fmt="{arg2}")
    def right(v1, v2):
        return v2

    # abs
    @tofunction()
    def sign(value):
        return np.sign(value)

    # neg
    @tofunction([Constant(-1)], fmt="(-{arg1})")
    def neg(value):
        return -value

    # abs
    @tofunction()
    def abs(value):
        return np.abs(value)

    # add
    @tofunction([Constant(1), Constant(1)], fmt="({arg1}+{arg2})")
    def add(v1, v2):
        return v1 + v2

    # sub
    @tofunction([Constant(1), Constant(-1)], fmt="({arg1}-{arg2})")
    def sub(v1, v2):
        return v1 - v2

    # mul
    @tofunction([right(p1, p2), left(p1, p2)], fmt="({arg1}*{arg2})")
    def mul(v1, v2):
        return v1 * v2

    # inv
    @tofunction(fmt="(1/{arg1})")
    def inv(value):
        return 1.0 / value

    # div
    @tofunction(fmt="({arg1}/{arg2})")
    def div(v1, v2):
        return v1 / v2

    # pow
    @tofunction(fmt="({arg1}**{arg2})")
    def pow(v1, v2):
        return v1**v2

    # log
    @tofunction()
    def log(value):
        return np.log(value)

    # exp
    @tofunction()
    def exp(value):
        return np.exp(value)

    # set missing derivatives
    abs.derivatives = [sign(p1)]
    inv.derivatives = [div(Constant(-1), pow(p1, Constant(2)))]
    div.derivatives = [inv(right(p1, p2)), div(neg(p1), pow(p2, Constant(2)))]
    pow.derivatives = [
        mul(p2, pow(p1, add(p2, Constant(-1)))),
        mul(log(p1), pow(p1, p2)),
    ]
    log.derivatives = [inv(p1)]
    exp.derivatives = [exp(p1)]


#
# utilities


def flatten(seq):
    """flatten nested list"""
    if not isinstance(seq, (list, tuple)):
        return [seq]
    return sum([flatten(item) for item in seq], start=[])
