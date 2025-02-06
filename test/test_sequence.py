import pytest
import numpy as np
from epgpy import sequence, operators as epgops

def test_sequence_class():
    from epgpy.sequence import operators, Sequence

    # operators
    necho = 5
    exc = operators.T(90, 90)
    rfc = operators.T(180, 0)
    spl = operators.S(1, duration=5)
    rlx = operators.E(5, 1400, 'T2')
    adc = operators.ADC

    # sequence 
    seq = Sequence([exc] + [spl, rlx, rfc, spl, rlx, adc] * necho)
    assert seq.variables == {"T2"}
    assert len(seq) == necho * 6 + 1
    assert list(seq) == seq.operators
    assert seq[-1] is seq.operators[-1]

    # insert 
    seq[1:1] = [adc] * 2
    assert seq[1:3] == [adc] * 2
    assert seq[3] == spl

    # delete
    seq[1:3] = []
    assert seq[1] == spl

    # build
    ops = seq.build({'T2': 3})
    assert all(isinstance(op, epgops.Operator) for op in ops)
    # check operators are not duplicated
    assert all(ops[3] is ops[3 + i * 6] for i in range(necho))
    assert all(ops[1] is ops[1 + i * 6] is ops[4 + i * 6] for i in range(necho))
    assert all(ops[2] is ops[2 + i * 6] is ops[5 + i * 6] for i in range(necho))
    
    # simulate
    signal = seq(T2=30)
    assert isinstance(signal, np.ndarray)
    assert signal.shape == (1, necho)

    # nd simulate
    sig = seq(T2=[10, 20, 30])
    assert isinstance(sig, np.ndarray)
    assert sig.shape == (3, necho)

    # gradient
    sig, jac = seq.jacobian(['T2', 'alpha'], T2=30)
    assert sig.shape == (1, necho) # shape x nadc
    assert jac.shape == (1, necho, 2) # shape x nadc x nvar

    1/0
    # hessian: add function to diff: `get_sequence_order2(seq, vars1, vars2=None)`

    # crlb

    # adc times
    # simulate options
    # string operators
    # edit, add sequences



def test_virtual_operator():
    from epgpy.sequence import operators

    # T
    T = operators.T('alpha', 'phi', name='T', duration=10)
    assert set(T.variables) == {'alpha', 'phi'}
    assert 'name' in T.options
    T = T(phi=90)
    assert set(T.variables) == {'alpha'}
    assert isinstance(T.alpha, sequence.Variable)
    assert isinstance(T.phi, sequence.Constant)
    assert T.name == 'T'
    assert T.duration == 10

    rf = T.build({'alpha': 90})
    assert isinstance(rf, epgops.T)
    assert rf.alpha == 90
    assert rf.phi == 90
    assert rf.name == 'T'
    assert rf.duration == 10

    # E
    E = operators.E('tau', 'T1', 'T2', g='freq', name='E', duration=0.0)
    assert set(E.variables) == {'tau', 'T1', 'T2', 'freq'}
    with pytest.raises(ValueError):
        rlx = E.build({'tau': 10, 'T1': 1e3, 'T2': 1e2})
    rlx = E.build({'tau': 10, 'T1': 1e3, 'T2': 1e2, 'freq': 5})
    assert isinstance(rlx, epgops.E)
    assert rlx.tau == 10
    assert rlx.T1 == 1e3
    assert rlx.T2 == 1e2
    assert rlx.g == 5


    # utilities
    Adc = operators.Adc(phase='phi', attr='Z0')
    assert set(Adc.variables) == {'phi'}
    assert Adc.attr == 'Z0'
    adc = Adc(phi=15).build()
    assert isinstance(adc, epgops.Adc)
    assert adc.phase == 15
    assert adc.attr == 'Z0'
    
    # ADC, RESET, etc.
    ADC = operators.ADC
    assert isinstance(ADC, sequence.VirtualOperator)
    assert ADC.operator is epgops.ADC
    



def test_expression():
    from epgpy.sequence import Constant, Variable, Proxy, functions, Expression, Function

    # constant and variables
    cst = Constant(2)
    var = Variable('var')
    assert cst.variables == set()
    assert var.variables == {'var'}
    assert var.variables == {Variable('var')}

    assert cst(var=10) == 2
    assert var(var=3.0) == 3.0

    assert cst.derive('anything', foo=1) == 0.0
    assert var.derive('anything', var=5.0) == 0.0
    assert var.derive('var', var=5.0) == 1.0

    # simple expression
    expr = cst + var
    assert isinstance(expr.function, Function)
    assert isinstance(expr.arguments[0], Constant)
    assert isinstance(expr.arguments[1], Variable)
    assert expr.variables == {var}

    # evaluate
    with pytest.raises(ValueError):
        expr() # missing variable
    with pytest.raises(ValueError):
        expr(var2=3.0) # wrong variable
    assert expr(var=3.0) == 5.0

    # proxy expression
    expr = Proxy(1) * cst
    assert expr.variables == {Proxy(1)}
    with pytest.raises(NotImplementedError):
        expr()
    expr = expr.map({Proxy(1): var})
    assert expr.variables == {var}
    assert expr(var=3) == 6.0 # 2 x 3

    # check operators and functions
    assert -var(var=3) == -3
    assert (cst + var)(var = 3) == cst.value + 3
    assert (cst - var)(var = 3) == cst.value - 3
    assert (cst * var)(var = 3) == cst.value * 3
    assert (cst / var)(var = 3) == cst.value / 3
    assert (cst ** var)(var = 3) == cst.value ** 3
    assert functions.log(var)(var=3) == np.log(3)
    assert functions.exp(var)(var=3) == np.exp(3)

    # derive
    expr = cst * var
    d_expr = expr.derive('var')
    assert isinstance(d_expr, Expression)
    assert d_expr.variables == expr.variables
    assert d_expr(var=3.0) == cst.value

    # check derivatives for operators and functions
    assert (-var).derive('var', var=3) == -1.0
    assert (cst + var).derive('var', var = 3) == 1.0
    assert (cst - var).derive('var', var = 3) == -1.0
    assert (cst * var).derive('var', var = 3) == cst.value
    assert (var / cst).derive('var', var = 3) == 1.0 / cst.value
    assert (cst / var).derive('var', var = 3) == cst.value * (-1 / 3**2)
    assert (var ** cst).derive('var', var = 3) == cst.value * 3 ** (cst.value - 1)
    assert (cst ** var).derive('var', var = 3) == np.log(cst.value) * cst.value ** 3
    assert functions.log(var).derive('var', var=3) == 1 / 3
    assert functions.exp(var).derive('var', var=3) == np.exp(3)

    # composed expression
    v1 = Variable('v1')
    v2 = Variable('v2')
    c1 = Constant(3)

    expr = functions.log(v1 / v2 - c1)
    assert expr(v1=12, v2=3) == 0.0
    assert expr.derive(v1)(v1=12, v2=3) == 1/3
    assert expr.derive(v2)(v1=12, v2=3) == -4/3

    # fix
    expr2 = expr.fix(v2=3)
    assert expr2.variables == {v1}
    assert expr(v1=12, v2=3) == expr2(v1=12)

    # ndarray
    c1 = Constant(np.linspace(-1, 1, 6).reshape(3, 2))
    expr = functions.exp(v1 / v2) + c1
    arr = np.arange(1, 7).reshape(3, 2)
    assert np.allclose(expr(v1=1, v2=arr), np.exp(1/arr) + c1.value)
    assert np.allclose(expr.derive('v1', v1=1, v2=arr), 1/arr * np.exp(1/arr))

    # convert
    expr = v1 * 'v2'
    assert isinstance(expr, Expression)
    assert expr.variables == {'v1', 'v2'}
    assert expr(v1=2, v2=3) == 2 * 3
    

    

