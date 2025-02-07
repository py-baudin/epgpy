import pytest
import numpy as np
from epgpy import sequence, operators as epgops

def test_sequence_class():
    from epgpy.sequence import operators, Sequence, Variable

    # variables
    T2 = Variable('T2')
    B1 = Variable('B1')

    # operators
    necho = 5
    exc = operators.T(90, 90)
    rfc = operators.T(180 * B1, 0)
    spl = operators.S(1, duration=5)
    rlx = operators.E(5, 1400, T2)
    adc = operators.ADC

    # sequence 
    seq = Sequence([exc] + [spl, rlx, rfc, spl, rlx, adc] * necho)
    assert seq.variables == {'T2', 'B1'}
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
    ops = seq.build({'T2': 3, 'B1': 1.0})
    assert all(isinstance(op, epgops.Operator) for op in ops)
    # check operators are not duplicated
    assert all(ops[3] is ops[3 + i * 6] for i in range(necho))
    assert all(ops[1] is ops[1 + i * 6] is ops[4 + i * 6] for i in range(necho))
    assert all(ops[2] is ops[2 + i * 6] is ops[5 + i * 6] for i in range(necho))
    
    # simulate
    sig = seq(T2=30, B1=1) # idem: seq.signal(T2=30, B1=1)
    assert isinstance(sig, np.ndarray)
    assert sig.shape == (1, necho)

    # nd simulate
    sig = seq.signal(T2=[10, 20, 30], B1=1.0)
    assert isinstance(sig, np.ndarray)
    assert sig.shape == (3, necho)

    # jacobians
    sig, jac = seq.jacobian(['T2', 'B1'], T2=30, B1=0.8)
    assert sig.shape == (1, necho) # shape x nadc
    assert jac.shape == (1, necho, 2) # shape x nadc x nvar
    # finite differences
    assert np.allclose(1e8 * (seq.signal(T2=30 + 1e-8, B1=0.8) - sig), jac[..., 0])
    assert np.allclose(1e8 * (seq.signal(T2=30, B1=0.8 + 1e-8) - sig), jac[..., 1])

    # hessian
    sig, jac, hes = seq.hessian(['T2', 'B1'], ['T2'], T2=30, B1=0.8)
    assert sig.shape == (1, necho) # shape x nadc
    assert jac.shape == (1, necho, 2) # shape x nadc x nvar
    assert hes.shape == (1, necho, 2, 1) # shape x nadc x nvar1 x nvar2
    # finite differences
    jacT2 = seq.jacobian('T2', T2=30 + 1e-8, B1=0.8)[1]
    jacB1 = seq.jacobian('B1', T2=30 + 1e-8, B1=0.8)[1]
    assert np.allclose(1e8 * (jacT2[..., 0] - jac[..., 0]), hes[..., 0, 0])
    assert np.allclose(1e8 * (jacB1[..., 0] - jac[..., 1]), hes[..., 1, 0])

    
    # crlb
    # adc times
    # simulate options
    # string operators
    # edit, add sequences

def test_sequence_multiple_variables():
    from epgpy.sequence import Variable, Sequence, operators

    necho = 5
    excit = operators.T(90, 90)
    refoc = operators.T("alpha", 0)
    shift = operators.S(1, duration=5)
    relax = operators.E(5, "T1", "T2")
    adc = operators.ADC

    # sequence object
    seq = Sequence([excit] + [shift, relax, refoc, relax, shift, adc] * necho)
    assert seq.variables == {"T2", "T1", "alpha"}

    # measure signal and derivative
    signal = seq.signal(T2=35, T1=1000, alpha=150)
    assert signal.shape == (1, necho)

    # dT2 = seq.derive("T2", T2=35, T1=1000, alpha=150)[..., 0]
    # dT1 = seq.derive("T1", T2=35, T1=1000, alpha=150)[..., 0]
    # dalpha = seq.derive("alpha", T2=35, T1=1000, alpha=150)[..., 0]

    # jacobian
    _, jac = seq.jacobian(["alpha", "T1", "T2"], T2=35, T1=1000, alpha=150)
    assert jac.shape == (1, necho, 3)

    # finite difference
    fdiff = (
        np.stack(
            [
                seq.signal(T2=35, T1=1000, alpha=150 + 1e-8),
                seq.signal(T2=35, T1=1000 + 1e-8, alpha=150),
                seq.signal(T2=35 + 1e-8, T1=1000, alpha=150),
            ],
            axis=-1,
        )
        - signal[..., np.newaxis]
    )
    assert np.allclose(fdiff * 1e8, jac, atol=1e-7)

    # hessian
    _, _, hess = seq.hessian(["alpha", "T1", "T2"], T2=35, T1=1000, alpha=150)
    assert hess.shape == (1, necho, 3, 3)

    fdiff = (
        np.stack(
            [
                np.stack(
                    [
                        seq.jacobian("alpha", T2=35, T1=1000, alpha=150 + 1e-8)[..., 0],
                        seq.jacobian("alpha", T2=35, T1=1000 + 1e-8, alpha=150)[..., 0],
                        seq.jacobian("alpha", T2=35 + 1e-8, T1=1000, alpha=150)[..., 0],
                    ],
                    axis=1,
                ),
                np.stack(
                    [
                        seq.jacobian("T1", T2=35, T1=1000, alpha=150 + 1e-8)[..., 0],
                        seq.jacobian("T1", T2=35, T1=1000 + 1e-8, alpha=150)[..., 0],
                        seq.jacobian("T1", T2=35 + 1e-8, T1=1000, alpha=150)[..., 0],
                    ],
                    axis=1,
                ),
                np.stack(
                    [
                        seq.derive("T2", T2=35, T1=1000, alpha=150 + 1e-8)[..., 0],
                        seq.derive("T2", T2=35, T1=1000 + 1e-8, alpha=150)[..., 0],
                        seq.derive("T2", T2=35 + 1e-8, T1=1000, alpha=150)[..., 0],
                    ],
                    axis=1,
                ),
            ],
            axis=2,
        )
        * 1e8
        - jac[..., np.newaxis, :] * 1e8
    )

    assert np.allclose(fdiff, hess, atol=1e-7)
    1/0

    # include magnitude as variable
    _, grad, hess = seq.hessian(["magnitude", "alpha"], T2=35, T1=1000, alpha=150)
    fdiff = (
        np.stack(
            [
                seq.signal(T2=35, T1=1000, alpha=150) * (1 + 1e-8) - signal,
                seq.signal(T2=35, T1=1000, alpha=150 + 1e-8) - signal,
            ],
            axis=-1,
        )
        * 1e8
    )
    assert np.allclose(grad, fdiff, atol=1e-7)

    fdiff = (
        np.stack(
            [
                np.concatenate(
                    [
                        seq.gradient("magnitude", T2=35, T1=1000, alpha=150)[1],
                        seq.gradient("magnitude", T2=35, T1=1000, alpha=150 + 1e-8)[1],
                    ],
                    axis=-1,
                ),
                np.concatenate(
                    [
                        seq.gradient("alpha", T2=35, T1=1000, alpha=150)[1]
                        * (1 + 1e-8),
                        seq.gradient("alpha", T2=35, T1=1000, alpha=150 + 1e-8)[1],
                    ],
                    axis=-1,
                ),
            ],
            axis=-1,
        )
        * 1e8
    )
    fdiff = fdiff - grad[..., np.newaxis, :] * 1e8
    assert np.allclose(hess, fdiff, atol=1e-7)

    # CRLB
    variables = ["alpha", "T2"]
    crlb, d_crlb = seq.crlb(variables, T2=35, T1=1000, alpha=150, gradient=True)
    fdiff = (
        np.stack(
            [
                seq.crlb(variables, T2=35, T1=1000, alpha=150 + 1e-8) - crlb,
                seq.crlb(variables, T2=35 + 1e-8, T1=1000, alpha=150) - crlb,
            ],
            axis=-1,
        )
        * 1e8
    )
    assert np.allclose(fdiff, d_crlb, rtol=1e-4)



def test_partial_hessian():
    from epgpy.sequence import Variable, Sequence, operators

    excit = operators.T(90, 90)
    refoc = operators.T("alpha", 0)
    shift = operators.S(1, duration=5)
    relax = operators.E(5, "T1", "T2")
    necho = 5
    adc = operators.ADC

    # sequence object
    seq = Sequence([excit] + [shift, relax, refoc, shift, relax, adc] * necho)

    # 2nd derivative for all 9 pairs
    # _, _, hess1 = seq.hessian(["alpha", "T1", "T2"], alpha=150, T1=1e3, T2=30)
    # assert hess1.shape == (1, 5, 3, 3)

    # 2nd derivatives for 2 selected pairs
    _, jac2, hess2 = seq.hessian(["T1", "T2"], ["alpha"], alpha=150, T1=1e3, T2=30)
    assert hess2.shape == (1, 5, 2, 1)
    # alpha-T1
    # assert np.allclose(hess1[..., 0, 1], hess2[..., 0, 0])
    # assert np.allclose(hess1[..., 1, 0], hess2[..., 0, 0])
    # assert np.allclose(hess1[..., 0, 2], hess2[..., 1, 0])
    # assert np.allclose(hess1[..., 2, 0], hess2[..., 1, 0])

    # crlb
    crlb1, cgrad1 = seq.crlb(["T1", "T2"], gradient=["alpha"], alpha=150, T1=1e3, T2=30)
    assert len(cgrad1) == 1

    crlb2, cgrad2 = sequence.stats.crlb(jac2, H=hess2)
    assert np.allclose(crlb1, crlb2)
    assert np.allclose(cgrad1, cgrad2)


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
    assert abs(var)(var=-3) == 3
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
    assert abs(var).derive('var', var=-3) == -1
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

    # unknown variable
    assert expr.derive('v3') == Constant(0)
    assert expr.derive('v3')(v1=1, v2=1) == 0

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
    
    expr = 'v1' * v2
    assert isinstance(expr, Expression)
    assert expr.variables == {'v1', 'v2'}
    assert expr(v1=2, v2=3) == 2 * 3
    

    

