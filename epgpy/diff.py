""" Differential operators """
import itertools
import abc
import logging
import numpy as np
from . import statematrix, operator, evolution, common

LOGGER = logging.getLogger(__name__)

""" TODO
- refactor derive1/2

"""


class DiffOperator(operator.Operator, abc.ABC):
    """ Differential Operator
    
    Computes partial derivatives of the state matrix
    
    """

    # parameters for which the differential operator exist
    PARAMETERS = []


    @abc.abstractmethod
    def _derive1(self, param, sm):
        """ TO IMPLEMENT """
        pass

    @abc.abstractmethod
    def _derive2(self, params, sm):
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
        coeffs1, coeffs2 = self._parse_partials(order1, order2)
        self.coeffs1 = coeffs1
        self.coeffs2 = coeffs2
        
        # compute inter operator derivatives automatically if `order2` was not provided in full
        self.auto_cross_derivatives = coeffs2 != order2
        super().__init__(*args, **kwargs)

    @property
    def parameters_order1(self):
        """Set of parameters used in 1st order derivatives"""
        return {param for var in self.coeffs1 for param in self.coeffs1[var]}

    @property
    def parameters_order2(self):
        """Set of parameters used in 2nd order derivatives"""
        return {
            param
            for v1, v2 in self.coeffs2
            for param in list(self.coeffs1.get(v1, []))
            + list(self.coeffs1.get(v2, []))
        }

    def derive1(self, param, sm):
        """apply 1st order differential operator w/r to parameter `param` """
        sm = self.prepare(sm, inplace=False)
        sm_d1 = self._derive1(param, sm)
        sm_d1._equilibrium *= 0  # remove equilibrium
        return sm_d1

    def derive2(self, params, sm):
        """apply 2nd order differential operator w/r to parameters pair `params` """
        sm = self.prepare(sm, inplace=False)
        sm_d2 = self._derive2(params, sm)
        sm_d2._equilibrium *= 0  # remove equilibrium
        return sm_d2

    def __call__(self, sm, *, inplace=False):
        """Apply operator to the state matrix"""

        order1 = getattr(sm, "order1", {})
        order2 = getattr(sm, "order2", {})

        order2 = self._apply_order2(sm, order1, order2) # inplace=inplace
        order1 = self._apply_order1(sm, order1) # inplace=inplace

        # update state matrix (derivatives are not copied)
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
        parameters = set(self.parameters)

        if isinstance(order1, str):
            # single variable
            order1 = [order1]

        if order1 == True:
            # action: compute magnetisation order1 w/r all operator's parameters
            # 1st derivatives of arguments are set to 1
            coeffs1.update({(self, param): {param: 1} for param in parameters})

        elif isinstance(order1, list):
            # derive with respect to operator's parameters
            invalid = set(order1) - parameters
            if invalid:
                raise ValueError(f"Invalid parameter(s): {invalid}")
            # 1st derivatives of arguments is set to 1
            coeffs1.update({(self, param): {param: 1} for param in order1})

        elif isinstance(order1, dict) and all(
            isinstance(order1, dict) for order1 in order1.values()
        ):
            # pass the 1st derivatives of the op's parameters w/r to a custom variable
            invalid = {
                param for order1 in order1.values() for param in order1
            } - parameters
            if invalid:
                raise ValueError(f"Invalid parameter(s): {invalid}")
            coeffs1.update({var: order1[var] for var in order1 if order1[var]})

        elif order1:
            raise ValueError(f"Invalid parameter 'order1' value: {order1}")

        # 2nd order derivatives
        # {(x, y): {arg1: d2 arg1/dxdy, arg2: d2 arg2/dxdy, ...}}

        parameters_pairs = get_combinations(parameters)

        if isinstance(order2, str):
            # single variable
            order2 = [(order2, order2)]

        elif isinstance(order2, list) and all(isinstance(var, str) for var in order2):
            # list of variables: derivatives w/r to all variable pairs
            order2 = list(get_combinations(order2))

        if order2 == True:
            # compute all 2nd order partial derivatives
            # update 1st order coeffs1 if not already set
            [coeffs1.setdefault((self, param), {param: 1}) for param in parameters]
            # assumes 2nd derivatives of arguments is 0
            coeffs2.update({((self, p1), (self, p2)): {} for p1, p2 in parameters_pairs})

        elif isinstance(order2, list) and all(isinstance(pair, tuple) for pair in order2):
            # compute *some* 2nd order partial derivatives
            invalid = set(order2) - parameters_pairs
            if invalid:
                raise ValueError(f"Invalid variables pair(s): {invalid}")
            # update 1st order coeffs1 if not already set
            [
                coeffs1.setdefault((self, param), {param: 1})
                for pair in order2
                for param in pair
            ]
            # assumes 2nd derivatives of arguments is 0
            coeffs2.update({((self, p1), (self, p2)): {} for p1, p2 in order2})

        elif isinstance(order2, dict) and all(
            isinstance(order2, dict) for order2 in order2.values()
        ):
            # compute the 2nd derivatives of the operator's parameters w/r to custom
            if not all(isinstance(pair, tuple) and len(pair) == 2 for pair in order2):
                raise ValueError(f"Invalid variable pair(s): {list(order2)}")

            invalid = {
                param for order2 in order2.values() for param in order2
            } - parameters
            if invalid:
                raise ValueError(f"Invalid parameter(s): {invalid}")
            # check 1st order partials
            missing = {
                var: set(order2[pair]) - set(coeffs1.get(var, []))
                for pair in order2
                for var in pair
            }
            if any(missing[var] for var in missing):
                raise ValueError(f"Missing 1st order derivatives for variables: {missing}")
            coeffs2.update(order2)

        elif order2:
            raise ValueError(f"Invalid parameter 'order2' value: {order2}")

        return coeffs1, coeffs2


    def _apply_order1(
        self, 
        sm,
        order1={},
        # inplace=False,
    ):
        """Apply 1st order derived operator"""
        # operator's partial derivatives for involved parameters
        parameters = {param for var in self.coeffs1 for param in self.coeffs1[var]}
        # apply operator to previous 1st-order partials
        order1_previous = {var: self(order1[var]) for var in order1}
        # apply derived opertors to previous element
        order1_partials = {param: self.derive1(param, sm) for param in parameters}
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
        # inplace=False,
    ):
        """Apply 2nd order derived operator"""
        # apply operator to previous 2nd order partials
        order2 = {frozenset(pair): value for pair, value in order2.items()}
        order2_previous = {pair: self(order2[pair]) for pair in order2}

        # 2nd derivatives of current operator
        gradient2 = {
            frozenset((v1, v2)): {
                (p1, p2): self.coeffs1[v1][p1] * self.coeffs1[v2][p2]
                for p1 in self.coeffs1[v1]
                for p2 in self.coeffs1[v2]
            }
            for v1, v2 in self.coeffs2
            if {v1, v2} <= set(self.coeffs1)
        }
        pairs = {pair for pairs in gradient2.values() for pair in pairs}
        order2_partials = {pair: self.derive2(pair, sm) for pair in pairs}
        order2_current = combine_partials(gradient2, order2_partials)

        # add non-zero 2nd order parameter derivatives (generally none)
        order2_params = {param for pair in self.coeffs2 for param in self.coeffs2[pair]}
        order2_params = combine_partials(
            {frozenset(pair): self.coeffs2[pair] for pair in self.coeffs2},
            {param: self.derive1(param, sm) for param in order2_params},
        )

        # cross derivatives
        variables_cross = {var for pair in self.coeffs2 for var in pair}
        order2_pairs = {frozenset(pair) for pair in self.coeffs2}
        if self.auto_cross_derivatives:
            # compute inter-operator derivatives for all variable pairs
            variables_cross |= {var for pair in order2 for var in pair}
            order2_pairs |= {frozenset((v1, v2)) for v1 in self.coeffs1 for v2 in order1}
        variables_previous = set(order1) & variables_cross
        variables_current = set(self.coeffs1) & variables_cross
        parameters = {param for var in variables_current for param in self.coeffs1[var]}
        order2_cross = {}
        for v2 in variables_previous:
            gradient12 = {
                frozenset((v1, v2)): self.coeffs1[v1]
                for v1 in variables_current
                if frozenset((v1, v2)) in order2_pairs
            }
            _partials = {param: self.derive1(param, order1[v2]) for param in parameters}
            _cross12 = combine_partials(gradient12, _partials)
            # repeat if it's twice the same variable
            _cross11 = {pair: _cross12[pair] for pair in _cross12 if len(pair) == 1}
            order2_cross = accumulate(order2_cross, _cross12, _cross11)

        # accumulate partial derivatives
        # if inplace:
        #     order2 = accumulate(order2, order2_params, order2_current, order2_cross)
        # else:
        order2 = accumulate(order2_previous, order2_params, order2_current, order2_cross)

        # store 2nd order partials as (v1, v2) and (v2, v1)
        for pair in list(order2):
            pair1 = tuple(pair)
            pair1 = pair1 + pair1 if len(pair1) == 1 else pair1
            pair2 = pair1[::-1]
            order2[pair1] = order2.pop(pair)
            order2[pair2] = order2[pair1]

        return order2



# Probe operators for Jacobian/Hessian

class Jacobian(operators.Probe):
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


class Hessian(operators.Probe):
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
                    hess = getattr(sm.order1.get(v2, sm.zeros), self.probe)
                elif "magnitude" == v2:
                    hess = getattr(sm.order1.get(v1, sm.zeros), self.probe)
                else:
                    hess = getattr(sm.order2.get((v1, v2), sm.zeros), self.probe)
                arrays[-1].append(hess)

        return common.asnumpy(arrays)  # copy


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

def get_combinations(seq1, seq2=None, sort=True):
    """return set of unique 2-combinations of 2 lists of items"""
    seq2 = seq1 if seq2 is None else seq2
    if sort:
        return {tuple(sorted(pair, key=str)) for pair in itertools.product(seq1, seq2)}
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
            part = partials[param] * variables[var][param]
            if not var in combined:
                combined[var] = part
            else:
                combined[var] += part
    return combined


def accumulate(main, *dicts, default=0):
    """update or accumulate dictionaries"""
    for other in dicts:
        for var in other:
            main.setdefault(var, default)
            main[var] += other[var]
    return main
