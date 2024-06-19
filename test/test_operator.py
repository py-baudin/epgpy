""" test epglib module """

import pytest
import numpy as np
from epgpy import statematrix, operator, common

common.DEFAULT_ARRAY_MODULE = "numpy"


class MyOp(operator.Operator):
    """custom operator"""

    def __init__(self, nshift=0, shape=[1], name=None, duration=None):
        self._nshift = nshift
        self._shape = tuple(shape)
        super().__init__(name=name, duration=duration)

    def _apply(self, sm):
        # broadcast states matrix
        shape = self.shape
        if self.ndim < sm.ndim:
            # add dimensions at the end
            shape = shape + (1,) * (sm.ndim - self.ndim)
        shape = np.broadcast_shapes(shape, sm.shape)
        if sm.shape == shape:
            # inplace update
            sm.states[:] = sm.states
        else:
            # no inplace
            sm.states = np.broadcast_to(sm.states, shape + sm.states.shape[-2:])

        return sm

    @property
    def nshift(self):
        return self._nshift

    @property
    def shape(self):
        return self._shape


def test_operator_class():
    op = operator.EmptyOperator()
    assert op.name == "EmptyOperator"
    assert op.duration == 0
    assert op.ndim == 1
    assert op.shape == (1,)
    assert op.nshift == 0

    op = operator.EmptyOperator(name="Foobar", duration=1)
    assert op.name == "Foobar"
    assert op.duration == 1

    # add
    op1 = operator.EmptyOperator(name="op1", duration=1)
    op2 = operator.EmptyOperator(name="op2", duration=2)
    op12 = op1 * op2
    assert isinstance(op12, operator.Operator)
    assert isinstance(op12, operator.MultiOperator)
    assert op12.operators == [op1, op2]

    # test custom operator
    op = MyOp(nshift=2, shape=(3, 4))
    assert op.nshift == 2
    assert op.shape == (3, 4)

    # checks apply
    sm = statematrix.StateMatrix(shape=(3, 1))

    # no broadcasting
    # inplace
    assert MyOp(shape=[1])(sm, inplace=True) is sm
    MyOp(shape=[1])(sm, inplace=True)
    assert sm.shape == (3, 1)
    MyOp(shape=[3, 1])(sm, inplace=True)
    assert sm.shape == (3, 1)

    # broadcasting
    # not inplace
    assert MyOp(shape=[1, 2])(sm, inplace=False).shape == (3, 2)
    assert sm.shape == (3, 1)
    # inplace
    MyOp(shape=[1, 2])(sm, inplace=True)
    assert sm.shape == (3, 2)

    # expand
    # not inplace
    assert MyOp(shape=[3, 2, 2])(sm, inplace=False).shape == (3, 2, 2)
    assert sm.shape == (3, 2)  # unchanged
    # inplace
    MyOp(shape=[3, 2, 2])(sm, inplace=True)
    assert sm.shape == (3, 2, 2)

    # incompatible shapes
    sm = statematrix.StateMatrix(shape=(3, 1))
    with pytest.raises(ValueError):
        MyOp(shape=[2, 1])(sm, inplace=False)
    with pytest.raises(ValueError):
        MyOp(shape=[4, 4])(sm, inplace=False)
    with pytest.raises(ValueError):
        MyOp(shape=[2, 3, 2])(sm, inplace=False)


def test_multiop_class():
    op1 = MyOp(name="op1", duration=1, shape=(1, 3), nshift=2)
    op2 = MyOp(name="op2", duration=2, shape=(2, 1), nshift=1)
    op12 = operator.MultiOperator([op1, op2], name="seq1")
    assert op12.operators == [op1, op2]
    assert op12.duration == 3
    assert op12.nshift == 3
    assert op12.shape == (2, 3)

    # using add
    assert op12.operators == (op1 * op2).operators

    # invalid combinations
    with pytest.raises(ValueError):
        MyOp(shape=(2, 1)) * MyOp(shape=(3,))

    with pytest.raises(ValueError):
        MyOp(shape=(2, 1)) * MyOp(shape=(3, 1))

    # apply
    sm = statematrix.StateMatrix(shape=(2, 3))
    assert op12(sm, inplace=True) is sm
    assert op12(statematrix.StateMatrix(shape=(2,)))
    assert op12(statematrix.StateMatrix(shape=(2, 3, 4)))

    with pytest.raises(ValueError):
        # invalid shape
        op12(statematrix.StateMatrix(shape=(3,)))


def test_combinable_class():
    class OP1(operator.CombinableOperator):
        def _apply(self, sm):
            return sm
        
        @classmethod
        def combinable(cls, other):
            return True
        
        @classmethod
        def combine(cls, ops):
            name = '|'.join(op.name for op in ops)
            duration = sum(op.duration for op in ops)
            return OP1(name=name, duration=duration)
        
    op1 = OP1(name='op1', duration=1)
    op2 = OP1(name='op2', duration=2)
    opnc = operator.EmptyOperator(name='opnc')

    opc = op1 @ op2
    assert opc.name == 'op1|op2'
    assert opc.duration == op1.duration + op2.duration

    with pytest.raises(TypeError):
        op1 @ opnc


def test_spoiler_class():
    sm0 = statematrix.StateMatrix(0.5 * np.ones((3, 3)))

    # spoiler
    sm1 = operator.SPOILER(sm0)
    assert np.allclose(sm0.F, 0.5)  # unchanged
    assert np.allclose(sm1.F, 0)
    assert np.allclose(sm1.Z, 0.5)  # unchanged


def test_reset_class():
    sm0 = statematrix.StateMatrix(0.5 * np.ones((3, 3)))

    # reset
    sm1 = operator.RESET(sm0)
    assert np.allclose(sm0.F, 0.5)  # unchanged
    assert np.allclose(sm1.states, sm1.equilibrium)
