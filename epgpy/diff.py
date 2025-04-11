"""Differential operators

Note: only works with linear/affine operators
"""

import itertools
import abc
import warnings
import logging
import numpy as np
from . import operator, common, probe

LOGGER = logging.getLogger(__name__)

""" TODO
- parallel partial calculation
"""


class DiffOperator(operator.Operator, abc.ABC):
    """Differential Operator

    Computes partial derivatives of the state matrix

    """

    # parameters for which the differential operator exist
    PARAMETERS_ORDER1 = set()
    PARAMETERS_ORDER2 = set()

    def _derive1(self, sm, param):
        """TO IMPLEMENT"""
        pass

    @abc.abstractmethod
    def _derive2(self, sm, params):
        """TO IMPLEMENT"""
        pass

    def __init__(self, *args, order1=False, order2=False, **kwargs):
        """Init differential operator

        Arguments:
            order1: # Specify 1st order partial derivatives to compute
                True/False: compute all or none of the partial derivative of the state matrix
                str <parameter name> (or list of): compute selected partial derivatives
                dict of str {<alias1>: <param1>, ...}
                    rename parameters and compute selected partial derivatives
                # dict of dict {<variable>: {<param1>: <coeff1>, ...}}:
                #    compute combined partial derivatives, using the coefficients of the 1st order parameter's derivatives

            order2: # Specify 2nd order partial derivatives to compute
                True/False:  compute all or none of the 2nd order partial derivative of the state matrix
                str <parameter name> (or list of): compute all 2nd order partial derivatives for selected variables
                (str, str) (<param1>, <param2>) (or list of): compute selected 2nd order partial derivatives
                # dict of dict {(<var1>, <var2>): {(<param1>, <param2): <coeff12>, ...}):
                #     compute combined partial derivatives, using the coefficients of the 2nd order parameter's derivatives
        """
        # set parameters for order1 and order2
        if "parameters_order1" in kwargs:
            self.PARAMETERS_ORDER1 = set(kwargs.pop("parameters_order1"))
        else:
            self.PARAMETERS_ORDER1 = set(self.PARAMETERS_ORDER1)

        if "parameters_order2" in kwargs:
            self.PARAMETERS_ORDER2 = {
                Pair(pair) for pair in kwargs.pop("parameters_order2")
            }
        else:
            self.PARAMETERS_ORDER2 = {Pair(pair) for pair in self.PARAMETERS_ORDER2}

        super().__init__(*args, **kwargs)

        # parse keywords
        self.order1, self.order2 = self._parse_partials(order1, order2)
        # activate auto cross derivatives if order2 was not passed in full
        self.auto_cross_derivatives = isinstance(order2, (bool, str)) or all(
            isinstance(item, str) for item in order2
        )

    @property
    def parameters_order1(self):
        return set(param for var in self.order1 for param in self.order1[var])

    @property
    def parameters_order2(self):
        """Pairs of parameters used in 2nd order derivatives"""
        return {
            Pair(p1, p2)
            for v1, v2 in self.order2
            for p1 in self.order1.get(v1, [])
            for p2 in self.order1.get(v2, [])
            # keep only valid parameter pairss
            if {(p1, p2), (p2, p1)} & self.PARAMETERS_ORDER2
        }

    def derive0(self, sm, inplace=False):
        """apply operator (without order1 and order2 differential operators)"""
        if not inplace:
            sm = sm.copy()
        return self._apply(sm)

    def derive1(self, sm, param, inplace=False):
        """apply 1st order differential operator w/r to parameter `param`"""
        if not inplace:
            sm = sm.copy()
        sm_d1 = self._derive1(sm, param)
        sm_d1.arrays.update("equilibrium", 0)  # remove equilibrium
        return sm_d1

    def derive2(self, sm, params, inplace=False):
        """apply 2nd order differential operator w/r to parameters pair `params`"""
        if not inplace:
            sm = sm.copy()
        sm_d2 = self._derive2(sm, Pair(params))
        sm_d2.arrays.update("equilibrium", 0)  # remove equilibrium
        return sm_d2

    def __call__(self, sm, *, inplace=False):
        """Apply operator to the state matrix"""

        order1 = getattr(sm, "order1", {})
        order2 = getattr(sm, "order2", {})

        # check and resize state matrix
        sm = self.prepare(sm, inplace=inplace)

        if order2 or self.order2:
            order2 = self._apply_order2(sm, order1, order2, inplace=inplace)
        if order1 or self.order1:
            order1 = self._apply_order1(sm, order1, inplace=inplace)

        # apply operator
        sm = self._apply(sm)

        # store derivatives
        sm.order1 = order1
        sm.order2 = order2
        return sm

    def copy(self, **kwargs):
        new = super().copy(**kwargs)
        new.PARAMETERS_ORDER1 = self.PARAMETERS_ORDER1
        new.PARAMETERS_ORDER2 = self.PARAMETERS_ORDER2
        new.order1 = self.order1.copy()
        new.order2 = self.order2.copy()
        new.auto_cross_derivatives = self.auto_cross_derivatives
        return new

    #
    # private

    def _parse_partials(self, order1=None, order2=None):
        """Parse order1 and order2 arguments"""

        # 1st order derivatives
        parameters = set(self.PARAMETERS_ORDER1)

        if (not order1) and isinstance(order2, (bool, str)):
            order1 = order2

        if isinstance(order1, str):
            # single variable
            order1 = [order1]

        if not order1:
            order1 = {}

        elif order1 is True:
            # action: compute magnetisation order1 w/r all operator's parameters
            order1 = {param: {param: 1} for param in parameters}

        elif isinstance(order1, list):
            # list of parameters
            order1 = {param: {param: 1} for param in order1}

        elif isinstance(order1, dict) and all(
            isinstance(value, str) for value in order1.values()
        ):
            # parameter aliases
            order1 = {var: {order1[var]: 1} for var in order1}

        elif isinstance(order1, dict) and all(
            isinstance(value, dict) for value in order1.values()
        ):
            # parameters derivatives for given variables
            pass

        else:
            raise ValueError(f"Invalid parameter 'order1' value: {order1}")

        # check parameters
        invalid = {param for var in order1 for param in set(order1[var]) - parameters}
        if invalid:
            raise ValueError(f"Unknown parameter(s): {invalid}")

        if not order2:
            return order1, set()

        if not order1:
            raise ValueError("order1 must be set.")

        # 2nd order derivatives
        if order2 == True:
            # compute all 2nd order partial derivatives
            order2 = {pair: {} for pair in self.PARAMETERS_ORDER2}

        elif isinstance(order2, str):
            # single variable
            order2 = {(order2, order2): {}}

        elif all(isinstance(param, str) for param in order2):
            # list of variables: derivatives w/r to all variable pairs
            order2 = {{Pair(pair): {}} for pair in get_combinations(order2)}

        elif not isinstance(order2, dict) and all(
            isinstance(pair, tuple) for pair in order2
        ):
            # compute *some* 2nd order partial derivatives
            order2 = {Pair(pair): {} for pair in order2}

        elif isinstance(order2, dict) and all(
            isinstance(pair, tuple) and isinstance(order2[pair], dict)
            for pair in order2
        ):
            # pass 2nd order partials of the parameters w/r variable pairs
            order2 = {Pair(pair): order2[pair] for pair in order2}

        elif order2:
            raise ValueError(f"Invalid parameter 'order2' value: {order2}")

        # check order2 parameters
        invalid = {pair for pair in order2 if not (set(pair) & set(order1))}
        if invalid:
            raise ValueError(
                f"Invalid variable pair(s), no match in order1 variables: {invalid}"
            )
        cross_vars = {pair for pair in order2 if (set(pair) - set(order1))}
        invalid = {pair for pair in cross_vars if order2[pair]}
        if invalid:
            raise ValueError(
                f"Invalid variable pair(s), expecting no coefficient: {invalid}"
            )
        invalid = {
            param for pair in order2 for param in (set(order2[pair]) - parameters)
        }
        if invalid:
            raise ValueError(f"Unknown parameter(s) in order2: {invalid}")
        param_pairs = {
            Pair(p1, p2)
            for v1, v2 in order2
            for p1 in order1.get(v1, [])
            for p2 in order1.get(v2, [])
        }

        # authorize invalid parameter pairs but warn user
        invalid = param_pairs - set(self.PARAMETERS_ORDER2)
        if invalid:
            warnings.warn(f"Invalid parameters pair(s) in {self}: {sorted(invalid)}")
            # raise ValueError(f"Invalid parameters pair(s) in {self}: {sorted(invalid)}")

        return order1, order2

    def _apply_order1(
        self,
        sm,
        order1={},
        derive0=None,
        derive1=None,
        inplace=False,
    ):
        """Apply 1st order derived operator"""
        derive0 = derive0 or self.derive0
        derive1 = derive1 or self.derive1

        # apply operator to previous 1st-order partials
        order1_previous = {var: derive0(order1[var], inplace=inplace) for var in order1}
        # apply derived opertors to previous element
        order1_partials = {
            param: derive1(sm, param, inplace=False) for param in self.parameters_order1
        }
        # combine partials
        order1_current = combine_partials(self.order1, order1_partials)

        # accumulate derivatives
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
        inplace=False,
    ):
        """Apply 2nd order derived operator"""
        derive0 = derive0 or self.derive0
        derive1 = derive1 or self.derive1
        derive2 = derive2 or self.derive2

        # remove duplicates
        order2 = {Pair(pair): order2[pair] for pair in order2}

        # apply operator to previous 2nd order partials
        order2_previous = {
            pair: derive0(order2[pair], inplace=inplace) for pair in order2
        }

        # 2nd order coefficient (often 0) and 1st derivative
        params_order1 = {param for pair in self.order2 for param in self.order2[pair]}
        order1_partials = {param: derive1(sm, param) for param in params_order1}
        order1_previous = combine_partials(self.order2, order1_partials)

        # 2nd derivatives of current operator
        order2_partials = {
            (p1, p2): derive2(sm, (p1, p2)) for p1, p2 in self.parameters_order2
        }
        order2_coeffs = {
            Pair(v1, v2): {
                Pair(p1, p2): c1 * c2
                for p1, c1 in self.order1.get(v1, {}).items()
                for p2, c2 in self.order1.get(v2, {}).items()
            }
            for v1, v2 in self.order2
        }
        order2_current = combine_partials(order2_coeffs, order2_partials)

        # cross derivatives
        if self.auto_cross_derivatives:
            vars_cross = {Pair(v1, v2) for v1 in self.order1 for v2 in order1}
        else:
            vars_cross = set(self.order2)
        params_cross = {
            param
            for pair in vars_cross
            for var in pair
            for param in self.order1.get(var, [])
        }

        order2_partials = {
            (param, var): derive1(order1[var], param)
            for var in order1
            for param in params_cross
        }
        order2_coeffs1 = {
            Pair(v1, v2): {(p1, v1): c1 for p1, c1 in self.order1[v2].items()}
            for v1 in order1
            for v2 in self.order1
            if (Pair(v1, v2) in vars_cross) and (v1 >= v2)
        }
        order2_cross1 = combine_partials(order2_coeffs1, order2_partials)
        order2_coeffs2 = {
            Pair(v1, v2): {(p1, v1): c1 for p1, c1 in self.order1[v2].items()}
            for v1 in order1
            for v2 in self.order1
            if (Pair(v1, v2) in vars_cross) and (v1 <= v2)
        }
        order2_cross2 = combine_partials(order2_coeffs2, order2_partials)

        # accumulate partial derivatives
        order2 = accumulate(
            order2_previous,
            order1_previous,
            order2_current,
            order2_cross1,
            order2_cross2,
        )

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
        xp = sm.array_module
        zeros = xp.zeros(sm.shape)
        _variables = [var for var in self.variables if var != "magnitude"]
        # retrieve jacobian arrays except for magnitude
        arrays = [
            getattr(sm.order1[var], self.probe) if var in sm.order1 else zeros
            for var in _variables
        ]
        if "magnitude" in self.variables:
            index = self.variables.index("magnitude")
            arrays.insert(index, getattr(sm, self.probe))
        # copy
        return common.asnumpy(xp.stack(arrays, axis=-1))


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
        xp = sm.array_module
        missing = xp.zeros(sm.shape)

        arrays = []
        for v1 in self.variables1:
            arrays.append([])
            for v2 in self.variables2:
                if "magnitude" == v1:
                    hess = (
                        getattr(sm.order1[v2], self.probe)
                        if v2 in sm.order1
                        else missing
                    )
                elif "magnitude" == v2:
                    hess = (
                        getattr(sm.order1[v1], self.probe)
                        if v1 in sm.order1
                        else missing
                    )
                else:
                    v12 = Pair(v1, v2)
                    hess = (
                        getattr(sm.order2[v12], self.probe)
                        if v12 in sm.order2
                        else missing
                    )

                arrays[-1].append(hess)
            arrays[-1] = xp.stack(arrays[-1], axis=-1)
        # copy
        return common.asnumpy(xp.stack(arrays, axis=-2))


class PartialsPruner:
    """callback functor to remove partials with negligible energy"""

    def __init__(self, *, condition=1e-5, variables=None):
        if callable(condition):
            self.condition = condition
        elif common.isscalar(condition):
            self.threshold = condition
            self.condition = self.test_norm
        else:
            raise TypeError(condition)
        self.variables = set(variables) if variables else None

    def test_norm(self, sm):
        return sm.norm < self.threshold

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
            if np.all(self.condition(order1[var])):
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
            if np.all(self.condition(order2[pair])):
                order2.pop(pair)


#
# utilities


def Pair(p1, p2=None):
    """return sorted pair"""
    if p2 is None:
        p1, p2 = p1
    if p1 > p2:
        return (p2, p1)
    return (p1, p2)


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
