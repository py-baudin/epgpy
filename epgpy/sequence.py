""" Sequence building tools

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
import numpy as np
from . import operators as epgops, functions as funcs


class Sequence:
    """ Sequence building object """

    def __init__(self, ops=[], *, name=None):
        """ Create Sequence from list of operators """
        ops = flatten(ops)
        ops = self.check(ops)
        self.operators = ops
        self.name = name

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
        else: # single insert
            op = self.check([op])[0]
            self.operators[item] = op
        # assume bulk insert
        self.operators[item] = ops

    def __delitem__(self, item):    
        del self.operators[item]

    def __add__(self, other):
        if not isinstance(other, Sequence):
            raise ValueError(f'Expecting Sequence, not: {type(other)}')
        return self.copy(self.operators + other.operators)
    
    def __repr__(self):
        return self.name if self.name else f"Sequence({len(self)})"
    
    def __call__(self, **kwargs):
        """ call signal """
        return self.signal(**kwargs)

    @property
    def variables(self):
        return {var for op in self.operators for var in op.variables}
    
    def check(self, ops):
        # replace string objects with virtual operators
        ops = [operators.STR_OPS.get(op, op) for op in ops]
        # check operator type
        invalid = {op for op in ops if not isinstance(op, VirtualOperator)}
        if invalid:
            raise ValueError(f'Invalid operator(s): {invalid}')
        return ops
    
    def copy(self, ops=None, **kwargs):
        ops = ops or self.operators
        name = kwargs.get('name', self.name)
        return Sequence(ops, name=name)

    def build(self, values, order1=None, order2=None):
        """ build EPG operators"""
        unique = {} # unique operators
        return [
            unique.setdefault(op, op.build(values, order1=order1))
            for op in self.operators
        ]
    
    def simulate(self, values, order1=None, order2=None, probe=None, **kwargs):
        """ return funcs.simulate """
        ops = self.build(values, order1=order1, order2=order2)
        sim = funcs.simulate(ops, probe=probe, **kwargs)
        return sim
    
    def signal(self, **values):
        """ return simulated signal as ndarray (... x nADC) """
        sim = self.simulate(values, asarray=True)
        return np.moveaxis(sim, 0, -1)
    
    def jacobian(self, variables, **values):
        """ return simulated signal as ndarray (... x nADC) """
        if isinstance(variables, str):
            variables = [variables]
        probe = [epgops.ADC, epgops.Jacobian(variables)]
        sim, grad = self.simulate(values, order1=variables, probe=probe, asarray=True)
        return np.moveaxis(sim, 0, -1), np.moveaxis(grad, [0, 1], [-2, -1])


class VirtualOperator(abc.ABC):
    """ Virtual operator base class """
    OPERATOR = None
    POSITIONALS = []
    KEYWORDS = []
    OPTIONS = []
    
    @property
    def variables(self):
        """ list of variables"""
        variables = []
        for var in self.positionals + list(self.keywords.values()): 
            if not isinstance(var, Variable):
                continue
            elif not str(var) in variables:
                variables.append(str(var))
        return variables
    
    def __init__(self, *args, **kwargs):
        # list positionals
        positionals = list(args) + [kwargs.pop(key) for key in set(kwargs) & set(self.POSITIONALS)]
        keywords = {key: kwargs.pop(key) for key in set(kwargs) & set(self.KEYWORDS)}
        options = kwargs
        # check options
        if not Ellipsis in self.OPTIONS:
            unknown = set(options) - set(self.OPTIONS)
            if unknown: raise ValueError(f'Unknown option(s): {options}')
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
        """ fix some values """
        args = [arg.fix(**values) for arg in self.positionals]
        keywords = {key: value.fix(**values) for key, value in self.keywords.items()}
        kwargs = {**keywords, **self.options}
        return type(self)(*args, **kwargs)

    def build(self, values={}, *, order1=None, order2=None):
        """ build (non-virtual) EPG operator """
        # solve expressions
        args = [arg(**values) for arg in self.positionals]
        keywords = {key: value(**values) for key, value in self.keywords.items()}
        kwargs = {**keywords, **self.options}
        # build operator
        if not (order1 or order2):
            return self.OPERATOR(*args, **kwargs)
        
        # build order1 and order2 dicts
        diff = {}
        for param in self.OPERATOR.PARAMETERS_ORDER1:
            if param in self.POSITIONALS:
                index = self.POSITIONALS.index(param)
                if index >= len(self.positionals):
                    var = Constant(0)
                else:
                    var = self.positionals[index]
            elif param in self.KEYWORDS:
                var = self.keywords[param]
            if not str(var) in order1:
                continue
            if isinstance(var, Constant):
                diff.setdefault('order1', {})[param] = param
            elif isinstance(var, Variable):
                diff.setdefault('order1', {})[str(var)] = param

        kwargs.update(diff)
        return self.OPERATOR(*args, **kwargs)
    
    def __repr__(self):
        args = ', '.join(repr(arg) for arg in self.positionals)
        return f'{self.OPERATOR.__name__}({args})'
    

class VirtualOperatorInstance(VirtualOperator):
    """ wrapper for ADC, SPOILER, etc."""
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
        raise NotImplementedError(f'Nothing to fix in {self}')

    def build(self, *args, **kwargs):
        return self.operator
    
    def __repr__(self):
        return repr(self.operator)
    

def virtual_operator(op, pos=[], kw=[], opt=[]):
    """ make virtual operator """
    if isinstance(op, epgops.Operator):
        return VirtualOperatorInstance(op, positionals=pos, options=opt)
    
    class Op(VirtualOperator):
        OPERATOR = op
        POSITIONALS = pos
        KEYWORDS = kw
        OPTIONS = opt
        def __init__(self, *args, **kwargs):
            """ place holder for docstring """
            super().__init__(*args, **kwargs)

    Op.__name__ = op.__name__
    Op.__doc__ = op.__doc__
    Op.__init__.__doc__ = op.__init__.__doc__
    return Op

    
class operators(types.SimpleNamespace):
    """availabel virtual operators """

    _std = ['name', 'duration']
    _diff = ['order1', 'order2']

    E = virtual_operator(epgops.E, ['tau', 'T1', 'T2', 'g'], [], _diff + _std)
    P = virtual_operator(epgops.P, ['g'], [], _diff + _std)
    R = virtual_operator(epgops.P, ['rT', 'rL', 'r0'], [], _diff + _std)
    T = virtual_operator(epgops.T, ['alpha','phi'], [], _diff + _std)
    Phi = virtual_operator(epgops.Phi, ['phi'], [], _diff + _std)
    S = virtual_operator(epgops.S, ['k'], [], _std)
    D = virtual_operator(epgops.D, ['tau', 'D', 'k'], [], _std)
    X = virtual_operator(epgops.X, ['tau', 'khi'], ['T1', 'T2', 'g'], _std)

    # utilities
    Adc = virtual_operator(epgops.Adc, [], ['phase'], ['attr', 'reduce', 'weights'] + _std)
    Wait = virtual_operator(epgops.Wait, ['duration'], [], ['name'])
    Offset = virtual_operator(epgops.Offset, ['duration'], [], ['name'])
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
        'ADC': ADC,
        'NULL': NULL,
        'SPOILER': SPOILER,
        'RESET': RESET,
    }


#
# Expressions

def to_expression(obj):
    """ return Expression """
    if isinstance(obj, Expression):
        return obj
    elif isinstance(obj, str):
        return Variable(obj)
    else:
        return Constant(obj)


class Expression:
    """ Mathematical expression """
    
    def __init__(self, function, arguments):
        self.function = function
        self.arguments = list(arguments)

    def __repr__(self):
        args = [repr(arg) for arg in self.arguments]
        return self.function.repr(args)

    def __call__(self, /, **values):
        """ compute expression """
        # solve arguments
        values = [arg(**values) for arg in self.arguments]
        # execute function
        return self.function.execute(*values)
    
    def fix(self, /, **values):
        """ transform variables into constants """
        arguments = [arg.fix(**values) for arg in self.arguments]
        return Expression(self.function, arguments)
    
    def map(self, mapping):
        """ map expression variables """
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
        """ compute derivative expression """
        variable = str(variable)
        d_expr = None
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
    
    def __sub__(self, other):
        other = to_expression(other)
        return Expression(functions.sub, [self, other])
    
    def __mul__(self, other):
        other = to_expression(other)
        return Expression(functions.mul, [self, other])
    
    def __truediv__(self, other):
        other = to_expression(other)
        return Expression(functions.div, [self, other])
    
    def __pow__(self, other):
        other = to_expression(other)
        return Expression(functions.pow, [self, other])
    

class Constant(Expression):
    """ A scalar constant """
    function = None
    arguments = []
    variables = set()

    def __init__(self, value, name=None):
        if isinstance(value, (np.ndarray, list)):
            value = np.asarray(value)
            name = name or f'arr[{", ".join(map(str, value.shape))}]'
        self.value = value
        self.name = name or f'{value}'

    def __repr__(self):
        return self.name

    def __call__(self, /, **kwargs):
        return self.value
    
    def fix(self, /, **kwargs):
        return self
    
    def derive(self, variable, /,  **kwargs):
        expr = Constant(0.0)
        return expr(**kwargs) if kwargs else expr
    

class Variable(Expression):
    """ A scalar variable """
    function = None
    arguments = []
    variables = set()

    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError(f'Expecting str, not {type(name)}')
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
            raise ValueError(f'Missing variable: {self.name}')
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
    """ proxy scalar argument """

    def __init__(self, position):
        if not isinstance(position, int):
            raise ValueError(f'Expecting int, not {type(position)}')
        self.position = position
        self.name = f'<arg{position}>'
        self.variables = {self}

    def __call__(self, /, **kwargs):
        raise NotImplementedError(f'Cannot solve a proxy variable')
    
    def derive(self, variable, /, **kwargs):
        raise NotImplementedError(f'Cannot derive a proxy variable')
        


class Function:
    """ Function wrapper including derivatives"""

    def __init__(self, function, *, derivatives=None, name=None, fmt=None, kwargs=None):
        if not callable(function):
            raise ValueError(f"Expecting callable, not {type(function)}")
        self.function = function
        self.kwargs = kwargs or {}
        self.name = name or function.__name__
        self.fmt = fmt or '{name}({args})'

        if derivatives:
            if not isinstance(derivatives, list):
                raise ValueError(f"Expecting list of derivatives, not {type(function)}")
            elif not all(isinstance(func, (type(None), Expression)) for func in derivatives):
                types = [type(func) for func in derivatives]
                raise ValueError(f"Expecting list of None or Function, not {types}")
        self.derivatives = derivatives

    def repr(self, args):
        strargs = {'args': ', '.join(args)}
        strargs.update({f'arg{i + 1}': arg for i, arg in enumerate(args)})
        return self.fmt.format(name=self.name, **strargs)

    def __repr__(self):
        return self.name

    def execute(self, *args):
        """ execute the function """
        return self.function(*args, **self.kwargs)
    
    def __call__(self, *args):
        """ return expression """
        args = [to_expression(arg) for arg in args]
        return Expression(self, args)

    def derive(self, i):
        """ return ith derivative expression """
        if not self.derivatives:
            raise ValueError(f'Undefined derivatives')
        expr = self.derivatives[i]
        if not expr:
            raise ValueError(f'Undefined {i}-th derivative')
        return expr


# helpers
def tofunction(derivatives=None, fmt=None):
    def wrapper(func):
        f = Function(func, derivatives=derivatives, fmt=fmt)
        return f
    return wrapper
    

class functions(types.SimpleNamespace):
    """ available functions """
    p1, p2 = Proxy(1), Proxy(2)

    # left and right
    @tofunction([Constant(1), Constant(0)], fmt='{arg1}')
    def left(v1, v2):
        return v1
    
    @tofunction([Constant(0), Constant(1)], fmt='{arg2}')
    def right(v1, v2):
        return v2

    # neg
    @tofunction([Constant(-1)], fmt='(-{arg1})')
    def neg(value):
        return - value

    # add
    @tofunction([Constant(1), Constant(1)], fmt='({arg1}+{arg2})')
    def add(v1, v2):
        return v1 + v2
    
    # sub
    @tofunction([Constant(1), Constant(-1)], fmt='({arg1}-{arg2})')
    def sub(v1, v2):
        return v1 - v2
  
    # mul
    @tofunction([right(p1, p2), left(p1, p2)], fmt='({arg1}*{arg2})')
    def mul(v1, v2):
        return v1 * v2
    
    # inv
    @tofunction(fmt='(1/{arg1})')
    def inv(value):
        return 1.0 / value

    # div
    @tofunction(fmt='({arg1}/{arg2})')
    def div(v1, v2):
        return v1 / v2

    # pow 
    @tofunction(fmt='({arg1}**{arg2})')
    def pow(v1, v2):
        return v1 ** v2

    # log
    @tofunction()
    def log(value):
        return np.log(value)
    
    # exp
    @tofunction()
    def exp(value):
        return np.exp(value)
    
    # set missing derivatives
    inv.derivatives = [div(Constant(-1), pow(p1, Constant(2)))]
    div.derivatives = [inv(right(p1, p2)), div(neg(p1), pow(p2, Constant(2)))]
    pow.derivatives = [mul(p2, pow(p1, add(p2, Constant(-1)))), mul(log(p1), pow(p1, p2))]
    log.derivatives = [inv(p1)]
    exp.derivatives = [exp(p1)]


#
# utilities

def flatten(seq):
    """ flatten nested list"""
    if not isinstance(seq, (list, tuple)):
        return [seq]
    return sum([flatten(item) for item in seq], start=[])