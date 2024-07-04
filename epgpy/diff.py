""" Differential operators 

Note: only works with linear/affine operators
"""
import itertools
import abc
import logging
import numpy as np
from . import operator, common, probe

LOGGER = logging.getLogger(__name__)

""" TODO
- compute 2nd derivatives only once!
- skip 2nd derivatives known to be 0

"""


class DiffOperator(operator.Operator, abc.ABC):
    """ Differential Operator 
    
    Computes partial derivatives of the state matrix
    
    """

    # parameters for which the differential operator exist
    PARAMETERS_ORDER1 = set()
    PARAMETERS_ORDER2 = set()

    def _derive1(self, sm, param):
        """ TO IMPLEMENT """
        pass

    @abc.abstractmethod
    def _derive2(self, sm, params):
        """ TO IMPLEMENT """
        pass


    def __init__(self, *args, order1=False, order2=False, **kwargs):
        """Init differential operator

        Arguments:
            order1: # Specify 1st order partial derivatives to compute
                True/False: compute all or none of the partial derivative of the state matrix
                str <parameter name> (or list of): compute selected partial derivatives
                dict of dict {<variable>: {<param1>: <coeff1>, ...}}: 
                    compute combined partial derivatives, using the coefficients of the 1st order parameter's derivatives

            order2: # Specify 2nd order partial derivatives to compute
                True/False:  compute all or none of the partial derivative of the state matrix
                str <parameter name> (or list of): compute all combinations of selected partial derivatives
                (str, str) (<param1>, <param2>) (or list of): : compute selected combinations of partial derivatives
                dict of dict {(<var1>, <var2>): {(<param1>, <param2): <coeff12>, ...}):
                    compute combined partial derivatives, using the coefficients of the 2nd order parameter's derivatives
        """
        # set parameters for order1 and order2
        if 'parameters_order1' in kwargs:
            self.PARAMETERS_ORDER1 = set( kwargs.pop('parameters_order1'))
        else:
            self.PARAMETERS_ORDER1 = set(self.PARAMETERS_ORDER1)

        if 'parameters_order2' in kwargs:
            self.PARAMETERS_ORDER2 = {Pair(pair) for pair in kwargs.pop('parameters_order2')}
        else:
            self.PARAMETERS_ORDER2 = {Pair(pair) for pair in self.PARAMETERS_ORDER2}

        super().__init__(*args, **kwargs)

        # compute coefficients
        coeffs1, coeffs2 = self._parse_partials(order1, order2)
        self.coeffs1 = coeffs1
        self.coeffs2 = coeffs2
        
        self.auto_cross_derivatives = False
        if isinstance(order2, (bool, str)) or all(isinstance(item, str) for item in order2):
            # compute inter operator derivatives automatically if `order2` pairs were not given
            self.auto_cross_derivatives = True

    @property
    def order1(self):
        return set(self.coeffs1)
    
    @property
    def order2(self):
        return {var for vars in self.coeffs2 for var in vars if var in self.coeffs1}

    @property
    def parameters_order1(self):
        """Set of parameters used in 1st order derivatives"""
        return {param for params in self.coeffs1.values() for param in params}

    @property
    def parameters_order2(self):
        """Pairs of parameters used in 2nd order derivatives"""
        return {
            Pair(p1, p2)
            for v1, v2 in self.coeffs2 
            for p1 in self.coeffs1.get(v1, [])
            for p2 in self.coeffs1.get(v2, [])
        } & self.PARAMETERS_ORDER2

    def derive0(self, sm):
        """ apply operator (without order1 and order2 differential operators)"""
        sm = self.prepare(sm, inplace=False)
        return self._apply(sm)

    def derive1(self, sm, param):
        """apply 1st order differential operator w/r to parameter `param` """
        sm = self.prepare(sm, inplace=False)
        sm_d1 = self._derive1(sm, param)
        sm_d1.arrays.set('equilibrium', [[0, 0, 0]], resize=True) # remove equilibrium
        return sm_d1

    def derive2(self, sm, params):
        """apply 2nd order differential operator w/r to parameters pair `params` """
        sm = self.prepare(sm, inplace=False)
        sm_d2 = self._derive2(sm, Pair(params))
        sm_d2.arrays.set('equilibrium', [[0, 0, 0]], resize=True) # remove equilibrium
        return sm_d2

    def __call__(self, sm, *, inplace=False):
        """Apply operator to the state matrix"""

        order1 = getattr(sm, "order1", {})
        order2 = getattr(sm, "order2", {})

        if order1 or self.coeffs1 or order2 or self.coeffs2:
            order2 = self._apply_order2(sm, order1, order2) # inplace=inplace
            order1 = self._apply_order1(sm, order1) # inplace=inplace

        # apply operator
        sm = super().__call__(sm, inplace=inplace)

        # store derivatives
        sm.order1 = order1
        sm.order2 = order2
        return sm

    # 
    # private

    def _parse_partials(self, order1=None, order2=None):
        """ Parse order1 and order2 arguments
        
        Parse simple structures used in the constructor to provided fully defined
        partial derivatives of relevant parameters, filling with default values where appropriate.

        Returned values are dict of dicts, containing the values of 1st or 2nd order derivatives
        of the operator's parameters with respect to the relevant independant variables.

        Often, the independant variables are the parameters themselves and the coefficients 
        are simply 1 (1st order) and 0 (2nd order): 
                `coeffs1 = {(op, 'p1'): {'p1': 1}, ...}`  
                `coeffs2 = {((op, 'p1'), (op, 'p1')): {('p1', 'p1'): 0}}`

        Return:
            coeffs1: values of the 1st order partial derivatives of the operator's parameters 
                with respect to the independant variables: {var1: {p1: d1/dvar1, ...}, ...}
            coeffs2:: values of the cross and 2nd order partial derivatives of the operator's parameters 
                with respect to independant variables: {(var1, var2): {p1: d2p1/dvar1/dvar2, ...}, ...}
        
        """
        coeffs1 = {}
        coeffs2 = {}

        # 1st order derivatives
        # {var1: {param1: dparam1/dvar1, param2: dparam2/dvar1}, var2: ...}
        parameters = set(self.PARAMETERS_ORDER1)

        if (not order1) and isinstance(order2, (bool, str)):
            order1 = order2

        if isinstance(order1, str):
            # single variable
            order1 = [order1]

        if order1 is True:
            # action: compute magnetisation order1 w/r all operator's parameters
            # 1st derivatives of arguments are set to 1
            coeffs1.update({param: {param: 1} for param in parameters})

        elif isinstance(order1, list):
            # derive with respect to operator's parameters
            invalid = set(order1) - parameters
            if invalid:
                raise ValueError(f"Unknown parameter(s): {invalid}")
            # 1st derivatives of variables is set to 1
            coeffs1.update({param: {param: 1} for param in order1})

        elif isinstance(order1, dict) and all(
            isinstance(coeffs, dict) for coeffs in order1.values()
        ):
            # pass the 1st derivatives of the op's parameters w/r to a custom variable
            invalid = {var for var in order1 if not set(order1[var]) <= parameters} 
            if invalid:
                raise ValueError(f"Unknown coefficients(s) in variable(s): {sorted(invalid)}")
            coeffs1.update({var: order1[var] for var in order1 if order1[var]})

        elif order1:
            raise ValueError(f"Invalid 'order1' value: {order1}")

        # 2nd order derivatives
        # {(x, y): {arg1: d2 arg1/dxdy, arg2: d2 arg2/dxdy, ...}}

        if order2 and not coeffs1:
            raise ValueError('order1 coefficients must be set.')

        # parameters_pairs = get_combinations(parameters)

        if isinstance(order2, str):
            # single variable
            order2 = [(order2, order2)]

        elif isinstance(order2, list) and all(isinstance(var, str) for var in order2):
            # list of variables: derivatives w/r to all variable pairs
            order2 = list(get_combinations(order2))

        if order2 == True:
            # compute all 2nd order partial derivatives
            # assumes 2nd derivatives of arguments is 0
            coeffs2.update({Pair(v1, v2): {} for v1, v2 in get_combinations(coeffs1)})

        elif isinstance(order2, list) and all(isinstance(pair, tuple) and len(pair)==2 for pair in order2):
            # compute *some* 2nd order partial derivatives
            invalid = {vars for vars in order2 if not set(vars) & set(coeffs1)}
            if invalid:
                raise ValueError(f'Unknown variable pair(s) in {self}: {sorted(invalid)}')
            coeffs2.update({Pair(v1, v2): {} for v1, v2 in order2})

        elif isinstance(order2, dict) and all(
            isinstance(pair, tuple) and len(pair)==2 and isinstance(coeffs, dict) for pair, coeffs in order2.items()
        ):
            # compute the 2nd derivatives of the operator's parameters w/r to custom
            invalid = {vars for vars in order2 if not set(vars) & set(order1)}
            if invalid:
                raise ValueError(f'Unknown variable pair(s): {sorted(invalid)}')
            invalid = {vars for vars in order2 if set(order2[vars]) - {param for var in order1 for param in order1[var]}}
            if invalid:
                raise ValueError(f'Missing 1st order derivatives for variable(s): {vars}')
            invalid = {vars for vars in order2 if set(order2[vars]) - parameters}
            if invalid:
                raise ValueError(f'Unknown coefficients in variable pair(s): {sorted(invalid)}')
            coeffs2.update({Pair(vars): order2[vars] for vars in order2})

        elif order2:
            raise ValueError(f"Invalid parameter 'order2' value: {order2}")

        return coeffs1, coeffs2


    def _apply_order1(
        self, 
        sm,
        order1={},
        derive0=None,
        derive1=None,
        # inplace=False,
    ):
        """Apply 1st order derived operator"""
        derive0 = derive0 or self.derive0 # self.__call__
        derive1 = derive1 or self.derive1

        # operator's partial derivatives for involved parameters
        parameters = {param for var in self.coeffs1 for param in self.coeffs1[var]}
        # apply operator to previous 1st-order partials
        order1_previous = {var: derive0(order1[var]) for var in order1}
        # apply derived opertors to previous element
        order1_partials = {param: derive1(sm, param) for param in parameters}
        # combine_partials partial derivatives
        order1_current = combine_partials(self.coeffs1, order1_partials)
        # if inplace:
        #     # breakpoint()
        #     order1 = accumulate(order1, order1_current)
        # else:
        order1 = accumulate(order1_previous, order1_current)

        return order1

    def _apply_order2(
        self,
        sm,
        order1={},
        order2={},
        derive0=None,
        derive1=None,
        derive2=None,
        # inplace=False,
    ):
        """Apply 2nd order derived operator"""
        derive0 = derive0 or self.derive0 
        derive1 = derive1 or self.derive1
        derive2 = derive2 or self.derive2

        # apply operator to previous 2nd order partials
        order2 = {Pair(pair): value for pair, value in order2.items()}
        order2_previous = {pair: derive0(order2[pair]) for pair in order2}

        # 2nd derivatives of current operator
        gradient2 = {
            Pair(v1, v2): {
                (p1, p2): self.coeffs1[v1][p1] * self.coeffs1[v2][p2]
                for p1 in self.coeffs1[v1]
                for p2 in self.coeffs1[v2]
            }
            for v1, v2 in self.coeffs2
            if {v1, v2} <= set(self.coeffs1)
        }
        # compute order 2 partials for valid parameter pairs
        pairs = {Pair(pair) for pairs in gradient2.values() for pair in pairs}
        order2_partials = {pair: derive2(sm, pair) for pair in pairs if pair in self.PARAMETERS_ORDER2}
        order2_partials.update({pair[::-1]: partial for pair, partial in order2_partials.items()})
        order2_current = combine_partials(gradient2, order2_partials)

        # add non-zero 2nd order parameter derivatives (generally none)
        order2_params = {param for pair in self.coeffs2 for param in self.coeffs2[pair]}
        order2_params = combine_partials(
            {Pair(pair): self.coeffs2[pair] for pair in self.coeffs2},
            {param: derive1(sm, param) for param in order2_params},
        )

        # cross derivatives
        variables_cross = {var for pair in self.coeffs2 for var in pair}
        order2_pairs = {Pair(pair) for pair in self.coeffs2}
        if self.auto_cross_derivatives:
            # compute inter-operator derivatives for all variable pairs
            # variables_cross |= {var for pair in order2 for var in pair}
            variables_cross |= set(order1)
            order2_pairs |= {Pair(v1, v2) for v1 in self.coeffs1 for v2 in order1}
        variables_previous = set(order1) & variables_cross
        variables_current = set(self.coeffs1) & variables_cross
        order1_params = {param for var in variables_current for param in self.coeffs1[var]}
        order2_cross = {}
        for v2 in variables_previous:
            gradient12 = {
                Pair(v1, v2): self.coeffs1[v1]
                for v1 in variables_current
                if Pair(v1, v2) in order2_pairs
            }
            _partials = {param: derive1(order1[v2], param) for param in order1_params}
            _cross12 = combine_partials(gradient12, _partials)
            # repeat if it's twice the same variable
            _cross11 = {pair: _cross12[pair] for pair in _cross12 if pair[0] == pair[1]}
            order2_cross = accumulate(order2_cross, _cross12, _cross11)

        # accumulate partial derivatives
        # if inplace:
        #     order2 = accumulate(order2, order2_params, order2_current, order2_cross)
        # else:
        order2 = accumulate(order2_previous, order2_params, order2_current, order2_cross)

        # store 2nd order partials as (v1, v2) and (v2, v1)
        for pair in list(order2):
            if pair[0] == pair[1]:
                continue
            order2[pair[::-1]] = order2[pair]

        return order2



# Probe operators for Jacobian/Hessian

class Jacobian(probe.Probe):
    """simplified probe operator for getting 1st derivatives"""

    def __init__(self, variables, *, probe="F0"):
        """
        parameters:
            variables: subset of variables to include in Jacobian (including "magnitude")
            probe: state matrix attribute (ie. F0, Z0, etc.)
        """
        self.probe = probe
        # setup variables
        if not isinstance(variables, list):
            variables = [variables]
        self.variables = variables

    def __repr__(self):
        return f"Jacobian({self.probe})"

    def _acquire(self, sm):
        """return signal's Jacobian"""
        _variables = [var for var in self.variables if var != "magnitude"]
        # retrieve jacobian arrays except for magnitude
        arrays = [
            getattr(sm.order1.get(var, sm.zeros), self.probe) for var in _variables
        ]
        if "magnitude" in self.variables:
            index = self.variables.index("magnitude")
            arrays.insert(index, getattr(sm, self.probe))
        return common.asnumpy(arrays)  # copy


class Hessian(probe.Probe):
    """simplified probe operator for getting 2nd derivatives"""

    def __init__(self, variables1, variables2=None, *, probe="F0"):
        """
        parameters:
            variables: subset of variables to include order2 (including "magnitude")
            probe: state matrix attribute (ie. F0, Z0, etc.)
        """
        self.probe = probe

        if not isinstance(variables1, list):
            variables1 = [variables1]

        if not variables2:
            variables2 = variables1
        elif not isinstance(variables2, list):
            variables2 = [variables2]

        self.variables1 = variables1
        self.variables2 = variables2

    def __repr__(self):
        return f"Hessian({self.probe})"

    def _acquire(self, sm):
        """return signal's Hessian"""
        arrays = []
        for v1 in self.variables1:
            arrays.append([])
            for v2 in self.variables2:
                if "magnitude" == v1:
                    # hess = getattr(sm.order1.get(v2, sm.zeros), self.probe)
                    hess = getattr(sm.order1[v2], self.probe) if v2 in sm.order1 else [0]
                elif "magnitude" == v2:
                    # hess = getattr(sm.order1.get(v1, sm.zeros), self.probe)
                    hess = getattr(sm.order1[v1], self.probe) if v2 in sm.order1 else [0]
                else:
                    #hess = getattr(sm.order2.get(Pair(v1, v2), sm.zeros), self.probe)
                    v12 = Pair(v1, v2)
                    hess = getattr(sm.order2[v12], self.probe) if v12 in sm.order2 else [0]
                    
                arrays[-1].append(hess)

        xp = sm.array_module
        return common.asnumpy(xp.stack(arrays))  # copy


class PartialsPruner:
    """callback functor to remove partials with negligible energy"""

    def __init__(self, variables=None, threshold=1e-5):
        self.threshold = threshold
        self.variables = set(variables) if variables else None

    def __repr__(self):
        if self.variables:
            return f"PartialsPruner({len(self.variables)} variables)"
        return "PartialsPruner(all variables)"

    def __call__(self, sm):
        order1 = getattr(sm, "order1", {})

        if not order1:
            return

        variables = set(order1)
        if self.variables:
            variables &= self.variables

        # order1
        for var in variables:
            if np.all(order1[var].norm < self.threshold):
                order1.pop(var)

        # order2
        order2 = getattr(sm, "order2", {})
        if not order2:
            return

        if self.variables:
            pairs = {pair for pair in order2 if set(pair) & self.variables}
        else:
            pairs = set(order2)

        for pair in pairs:
            if np.all(order2[pair].norm < self.threshold):
                order2.pop(pair)
        




#
# utilities

def Pair(*args):
    """ return sorted tuple"""
    if len(args) == 1:
        assert len(args[0]) == 2, 'A pair must have 2 values'
        return tuple(sorted(args[0]))
    elif len(args) == 2:
        return tuple(sorted(args))
    else:
        raise AssertionError('A pair musst have 2 values')

def get_combinations(seq1, seq2=None, sort=True):
    """return set of unique 2-combinations of 2 lists of items"""
    seq2 = seq1 if seq2 is None else seq2
    if sort:
        return {Pair(pair) for pair in itertools.product(seq1, seq2)}
    unique = set()
    return {
        pair
        for pair in itertools.product(seq1, seq2)
        if not (pair in unique or (unique.add(pair) or unique.add(pair[::-1])))
    }


def combine_partials(variables, partials):
    """combine partial derivatives"""
    combined = {}
    for var in variables:
        for param in variables[var]:
            if not param in partials:
                continue
            part = partials[param] * variables[var][param]
            if not var in combined:
                combined[var] = part
            else:
                combined[var] += part
    return combined


def accumulate(dict1, *dicts):
    """accumulate values from multiple dictionaries"""
    for other in dicts:
        for var in other:
            if not var in dict1:
                dict1[var] = other[var]
            else:
                dict1[var] += other[var]
    return dict1
