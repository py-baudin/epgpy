import time
import numpy as np
import epgpy as epg
from epgpy import diff
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    # define MRF sequence
    nTR = 400
    T1, T2 = 1380, 80

    alphas = [f"alpha_{i:03d}" for i in range(nTR)]
    taus = [f"tau_{i:03d}" for i in range(nTR)]
    order2_rf = [[("T1", alphas[i]), ("T2", alphas[i])] for i in range(nTR)]
    order2_rlx = [[("T1", taus[i]), ("T2", taus[i])] for i in range(nTR)]

    def rf(i, alpha):
        return epg.T(
            alpha,
            90,
            order1={alphas[i]: "alpha"},  # use parameter alias
            order2=order2_rf[i],  # select cross derivatives
        )

    def rlx(i, tau):
        return epg.E(
            tau,
            T1,
            T2,
            order1={"T1": "T1", "T2": "T2", taus[i]: "tau"},  # use parameter aliases
            order2=sum(order2_rlx + order2_rf, start=[]),  # select cross derivatives
            duration=True,
        )

    grd = epg.S(1)
    adc = epg.ADC

    def sequence(angles, times):
        """MRF sequence"""
        return [[rf(i, angles[i]), rlx(i, times[i]), adc, grd] for i in range(nTR)]

    # simulate with order2 derivatives
    angles = np.random.uniform(10, 60, nTR)
    times = np.random.uniform(11, 16, nTR)

    Jac = epg.Jacobian(["magnitude", "T1", "T2"])
    Hes = epg.Hessian(["magnitude", "T1", "T2"], alphas + taus)

    print(f"Simulate MRF sequence (nTR={nTR})")
    tic = time.time()
    hes = epg.simulate(
        sequence(angles, times),
        probe=Hes,
        max_nstate=10,
        disp=True,
        parallel=True,
    )

    toc = time.time()
    print(f"Done. " f"Duration: {toc - tic:.1f}s, ")
