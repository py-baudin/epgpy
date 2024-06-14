""" base operator classes """
# Python core imports
import abc
import logging

import numpy as np
from . import statematrix, common

LOGGER = logging.getLogger(__name__)


class Operator(abc.ABC):
    """Base operator class

    This class needs to be inherited, with the _apply method overridden

    Pre-implemented methods:
        ```Op1 * Op2```: returns a MultiOperator object made of Op1 and Op2
        ```Op(state_matrix)```: apply operator to the state matrix
    """

    def __init__(self, *, name=None, duration=None):
        """Create a generic operator

        Args:
            name: set operator name (defaults to the class name)
            duration: set operator duration (default 0)
                Use it if you need to calculate the sequence timing.

        """
        if duration is None:
            duration = 0
        elif np.any(np.asarray(duration) < 0):
            raise ValueError("Cannot have duration < 0")
        self.duration = duration
        self.name = name if name else type(self).__name__

    @abc.abstractmethod
    def _apply(self, sm: statematrix.StateMatrix) -> statematrix.StateMatrix:
        """does nothing: TO IMPLEMENT"""
        return sm


    @classmethod
    def from_list(cls, sequence):
        """create new operator from a sequence of operators"""
        return MultiOperator(sequence)

    @property
    def shape(self):
        return (1,)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def nshift(self):
        return 0

    def __repr__(self):
        """show operator name"""
        return self.name

    def __mul__(self, other):
        """return MultiOperator operator"""
        return Operator.from_list([self, other])
    
    def prepare(self, sm, inplace=False):
        """ check, resize and copy state matrix"""

        if not isinstance(sm, statematrix.StateMatrix):
            # check type
            raise TypeError(f"Not a StateMatrix: {sm}")

        elif not common.broadcastable(sm.shape, self.shape, append=True):
            # check shapes
            raise ValueError(
                f"Incompatible StateMatrix and operator shapes: {sm.shape}, {self.shape}"
            )
        
        # make new state matrix ?
        if not inplace or not sm.writeable:
            sm = sm.copy()

        if sm.ndim < self.ndim:
            # check ndims
            sm.expand(self.ndim)

        return sm    

    def __call__(self, sm, *, inplace=False):
        """apply operator"""

        # check and resize state matrix
        sm = self.prepare(sm, inplace=inplace)

        # apply
        sm = self._apply(sm)
        return sm
    
    # def copy(self, name=None, duration=None, **kwargs):
    #     """return copy of self"""
    #     raise NotImplementedError()

    # def combinable(self, other):
    #     """check if `other` can be combined with self"""
    #     raise NotImplementedError()

    # @staticmethod
    # def combine(self, ops, name=None, duration=None):
    #     """combine multiple operators """
    #     raise NotImplementedError()
        


#
# MultiOperator
class MultiOperator(Operator):
    """An operator made of a sequence of operators"""

    def __init__(self, operators=None, *, name=None, duration=None):
        """Create an operator from a sequence of operators

        Args:
            operators: seq
                Any sequence-type object containing operators
            name: see Operator
                Default: "Op1.name | Op2.name | ..."
            cf. Operator for remaining keyword arguments

        """

        # init sequence if necessary
        operators = [] if not operators else list(operators)

        self._nshift = 0
        self._ndim = 1
        self._shape = (1,)

        # list of operators
        self.operators = []
        self.duration = 0
        for op in operators:
            self.append(op)

        if not name:  # default name
            name = " | ".join([op.name for op in operators])
        if duration is None:  # use sum of durations
            duration = self.duration

        # init parent class
        super().__init__(name=name, duration=duration)

    def _apply(self, sm):
        """apply sequence of operators to state matrix"""
        for op in self.operators:
            # skip checks
            sm = op._apply(sm)
        return sm

    @property
    def shape(self):
        return self._shape

    @property
    def nshift(self):
        return self._nshift

    def __iter__(self):
        """iterate through object"""
        return iter(self.operators)

    def __len__(self):
        """sequence's length"""
        return len(self.operators)

    def __getitem__(self, i):
        """get i-th item"""
        return self.operators[i]

    def __mul__(self, other):
        self.append(other)
        return self

    def append(self, op):
        """add a new operator to the existing list"""

        if not isinstance(op, Operator):
            raise TypeError("Invalid operator: %s" % str(op))

        # common shape
        shape = common.broadcast_shapes(self.shape, op.shape, append=True)

        if isinstance(op, MultiOperator):
            # extend sequence if already a MultiOperatir
            self.operators.extend(op.operators)
        else:
            # append operator to sequence
            self.operators.append(op)

        self._shape = shape
        self._nshift += op.nshift
        self.duration += op.duration

    # def combinable(self, other):
    #     return self.operators[-1].combinable(other)

    # def combine(self, other, *others, name=None, duration=None):
    #     """call combine on last operator"""
    #     last = self.operators[-1].combine(other, *others)
    #     return MultiOperator(self.operators[:-1] + [last], name=name, duration=duration)


#
# Empty operator


class EmptyOperator(Operator):
    """Empty operator: does nothing"""

    def _apply(self, sm):
        return sm


# Empty operator instance: does nothing
NULL = EmptyOperator(name="NULL")


class Wait(EmptyOperator):
    """Empty operator with given duration"""

    def __init__(self, duration, name=None):
        """Init wait operator"""
        name = name if name is not None else f"Wait({duration})"
        super().__init__(duration=duration, name=name)


class Offset(EmptyOperator):
    """Empty operator with possibly negative duration"""

    def __init__(self, duration, name=None):
        name = name if name is not None else f"Offset({duration})"
        super().__init__(duration=abs(duration), name=name)
        self.duration = duration


#
# Spoiler


class Spoiler(Operator):
    """Perfect spoiler: destroy transverse magnetization"""

    def _apply(self, sm):
        sm.states[..., 0:2] = 0
        return sm


# Spoiler instance
SPOILER = Spoiler(name="Spoiler")


#
# Reset


class Reset(Operator):
    """Reset magnetization (return to equilibrium)"""

    def _apply(self, sm):
        return sm.copy(sm.equilibrium)


# Spoiler instance
RESET = Reset(name="Reset")
