import numpy as np

from . import common


def crlb(J, H=None, *, W=None, sigma2=1, log=False):
    """Cramer-Rao lower bound cost function"""
    xp = common.get_array_module(J)

    # J.shape: ... x npoint x nparam
    J = xp.asarray(J)

    # Fisher information  matrix
    I = 1 / sigma2 * xp.einsum("...np,...nq->...pq", J.conj(), J).real

    is_singular = np.linalg.cond(I) > 1e30
    I[is_singular] = np.nan
    lb = xp.linalg.inv(I)

    if W is not None:  # apply weights
        W = xp.asarray(W)[..., np.newaxis]
    else:
        W = 1
    cost = xp.trace(W * lb, axis1=-2, axis2=-1)

    if H is None:
        # return CRLB
        return cost if not log else np.log10(cost)

    # return CRLB and its gradient
    HJ = xp.einsum("...npx,...nq->...qpx", H.conj(), J) * 1 / sigma2
    HJ += np.moveaxis(HJ, -3, -2).conj()
    grad = -xp.einsum("...pq,...qrx,...rp->...x", W * lb, HJ.real, lb)
    if not log:
        return cost, grad
    return np.log10(cost), grad / cost[..., np.newaxis] / np.log(10)


def crlb_split(J, W=None, sigma2=1, log=False):
    """CRB for each variables in Jacobian"""
    xp = common.get_array_module(J)
    J = xp.asarray(J)
    I = 1 / sigma2 * xp.einsum("...np,...nq->...pq", J.conj(), J).real
    is_singular = np.linalg.cond(I) > 1e30
    I[is_singular] = np.nan
    lb = xp.linalg.inv(I)

    idiag = xp.arange(lb.shape[-1])
    crb = lb[..., idiag, idiag]
    if W is not None:  # apply weights
        crb *= xp.asarray(W)
    if log:
        crb = np.log10(crb)
    return xp.moveaxis(crb, -1, 0)


def confint(obs, pred, jac, hess=None, *, conflevel=0.95):
    """Delta method for confidence intervals and confidence bands"""
    nobs, nparam = jac.shape[-2:]
    dof = nobs - nparam
    res = obs - pred
    sse = np.sum(res * res.conj(), axis=-1).real

    # covariance matrix
    if hess is not None:
        # hessian of MLE
        Hmle = np.einsum("...nqp,...y->...pq", hess.conj(), res).real
        Hmle += np.einsum("...np,...nq->...pq", jac.conj(), jac).real
        cov = np.linalg.inv(Hmle)
    else:
        jac2 = np.einsum("...np,...nq->...pq", jac.conj(), jac).real
        cov = np.linalg.inv(jac2)
    cov *= sse[..., np.newaxis, np.newaxis] / dof

    # tvalue
    tval = get_tstat_interval(conflevel, dof)

    # confidence intervals for the parameters
    idiag = np.arange(nparam)
    cints = tval * np.sqrt(cov[..., idiag, idiag])

    # confidence band for the prediction
    predvar = np.einsum("...np,...pq,...nq->...n", jac.conj(), cov, jac).real
    cband = tval * np.sqrt(predvar)

    return cints, cband


#
# t confidence interval


def get_tstat_interval(alpha, nu):
    global TSTAT_INTERVAL
    key = (alpha, nu)
    if not key in TSTAT_INTERVAL:
        print("Importing scipy.stats")
        from scipy import stats

        TSTAT_INTERVAL[key] = stats.t.interval(alpha, nu)[1]
    return TSTAT_INTERVAL[key]


TSTAT_INTERVAL = {
    (0.95, 1): 12.706204736432095,
    (0.95, 2): 4.302652729911275,
    (0.95, 3): 3.182446305284263,
    (0.95, 4): 2.7764451051977987,
    (0.95, 5): 2.5705818366147395,
    (0.95, 6): 2.4469118487916806,
    (0.95, 7): 2.3646242510102993,
    (0.95, 8): 2.3060041350333704,
    (0.95, 9): 2.2621571627409915,
    (0.99, 1): 63.65674116287399,
    (0.99, 2): 9.92484320091807,
    (0.99, 3): 5.84090929975643,
    (0.99, 4): 4.604094871415897,
    (0.99, 5): 4.032142983557536,
    (0.99, 6): 3.707428021324907,
    (0.99, 7): 3.4994832973505026,
    (0.99, 8): 3.3553873313333957,
    (0.99, 9): 3.2498355440153697,
    (0.95, 10): 2.2281388519649385,
    (0.95, 11): 2.200985160082949,
    (0.95, 12): 2.1788128296634177,
    (0.95, 13): 2.1603686564610127,
    (0.95, 14): 2.1447866879169273,
    (0.95, 15): 2.131449545559323,
    (0.95, 16): 2.1199052992210112,
    (0.95, 17): 2.1098155778331806,
    (0.95, 18): 2.10092204024096,
    (0.95, 19): 2.093024054408263,
    (0.95, 20): 2.0859634472658364,
    (0.95, 21): 2.079613844727662,
    (0.95, 22): 2.0738730679040147,
    (0.95, 23): 2.0686576104190406,
    (0.95, 24): 2.0638985616280205,
    (0.95, 25): 2.059538552753294,
    (0.95, 26): 2.055529438642871,
    (0.95, 27): 2.0518305164802833,
    (0.95, 28): 2.048407141795244,
    (0.95, 29): 2.045229642132703,
    (0.95, 30): 2.0422724563012373,
    (0.95, 31): 2.0395134463964077,
    (0.95, 32): 2.036933343460101,
    (0.95, 33): 2.0345152974493383,
    (0.95, 34): 2.032244509317718,
    (0.95, 35): 2.0301079282503425,
    (0.95, 36): 2.0280940009804502,
    (0.95, 37): 2.0261924630291093,
    (0.95, 38): 2.024394164575136,
    (0.95, 39): 2.022690911734728,
    (0.95, 40): 2.0210753829953374,
    (0.95, 41): 2.0195409639828936,
    (0.95, 42): 2.018081697095881,
    (0.95, 43): 2.0166921941428133,
    (0.95, 44): 2.015367569912941,
    (0.95, 45): 2.0141033848332923,
    (0.95, 46): 2.0128955952945886,
    (0.95, 47): 2.0117405104757546,
    (0.95, 48): 2.0106347546964454,
    (0.95, 49): 2.009575234489209,
    (0.95, 50): 2.008559109715206,
    (0.95, 51): 2.007583768155882,
    (0.95, 52): 2.0066468031022113,
    (0.95, 53): 2.0057459935369497,
    (0.95, 54): 2.004879286566523,
    (0.95, 55): 2.004044781810181,
    (0.95, 56): 2.0032407174966975,
    (0.95, 57): 2.0024654580545986,
    (0.95, 58): 2.0017174830120923,
    (0.95, 59): 2.00099537704821,
    (0.95, 60): 2.0002978210582616,
    (0.95, 61): 1.9996235841149779,
    (0.95, 62): 1.9989715162223112,
    (0.95, 63): 1.9983405417721956,
    (0.95, 64): 1.9977296536259734,
    (0.95, 65): 1.9971379077520122,
    (0.95, 66): 1.9965644183594744,
    (0.95, 67): 1.9960083534755055,
    (0.95, 68): 1.9954689309194018,
    (0.95, 69): 1.9949454146328136,
    (0.95, 70): 1.9944371113297727,
    (0.95, 71): 1.993943367434504,
    (0.95, 72): 1.9934635662785827,
    (0.95, 73): 1.9929971255321663,
    (0.95, 74): 1.99254349484682,
    (0.95, 75): 1.9921021536898653,
    (0.95, 76): 1.9916726093523487,
    (0.95, 77): 1.9912543951146038,
    (0.95, 78): 1.990847068555052,
    (0.95, 79): 1.9904502099893602,
    (0.95, 80): 1.990063421028384,
    (0.95, 81): 1.9896863232444828,
    (0.95, 82): 1.9893185569368186,
    (0.95, 83): 1.988959779987179,
    (0.95, 84): 1.9886096667986732,
    (0.95, 85): 1.9882679073103775,
    (0.95, 86): 1.9879342060816718,
    (0.95, 87): 1.9876082814405769,
    (0.95, 88): 1.9872898646909385,
    (0.95, 89): 1.9869786993737677,
    (0.95, 90): 1.9866745405784678,
    (0.95, 91): 1.9863771543000648,
    (0.95, 92): 1.9860863168388934,
    (0.95, 93): 1.9858018142395026,
    (0.95, 94): 1.9855234417658298,
    (0.95, 95): 1.9852510034099262,
    (0.95, 96): 1.984984311431769,
    (0.95, 97): 1.984723185927883,
    (0.95, 98): 1.984467454426692,
    (0.95, 99): 1.9842169515086827,
}
