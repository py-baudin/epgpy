"""
Steady State and Transient Behavior of Gradient-Echo Sequences

from:
    Malik SJ, Teixeira RPAG, Hajnal JV: 
    Extended phase graph formalism for systems with magnetization transfer and exchange: 
    Magn Reson Med 2018; 80:767–779.

"""

import numpy as np
from matplotlib import pyplot as plt

from epgpy import operators, functions, statematrix, exchange, magnettransfer

# single component model
model0 = {
    "T1": [779],
    "T2": [45],
    "f": [1],
    "kmat": 0,
}

# Myelin water-exchange model
model1 = {
    "T1": [1000, 500],  # ms
    "T2": [100, 20],  # ms
    "khi": 2e-3,  # 1/ms
    "f": [1 - 0.2, 0.2],  # densities
}
model1["kmat"] = exchange.exchange_matrix(model1["khi"], densities=model1["f"])

# White matter MT model
model2 = {
    "T1": [779, 779],  # ms
    "T2": [45, 12e-3],  # ms
    "khi": 4.3e-3,  # 1/ms
    "f": [1 - 0.117, 0.117],  # densities
}
model2["kmat"] = exchange.exchange_matrix(model2["khi"], densities=model2["f"])

# mean residence time
# mrt = lambda f, khi: 1e3 * f / (khi * (1 - f))

# sequences
Nrf = 200
MaxPhi = 180
FA = 10
PH = np.linspace(1, 180, 2 * 179 + 1)
i117 = np.argmin(np.abs(PH - 117))
TR = 5  # ms

# MT
b1 = 13  # uT
G = 15.1 * 1e-3  # ms
gamma = 267.5221 * 1e-3  # rad /ms /uT
trf = (np.pi / 180 * FA) / (gamma * b1)  # ms
W = magnettransfer.saturation_rate(trf, b1, G)  # 1/ms
# W = np.pi * gamma ** 2 * G * b1 ** 2


# operators
adc = operators.Adc(reduce=0)  # sum both pools
shift = operators.S(1)
rx = operators.E(TR, model0["T1"][0], model0["T2"][0])
exg = operators.X(TR, model1["kmat"], T1=model1["T1"], T2=model1["T2"])
mt = operators.X(TR, model2["kmat"], T1=model2["T1"], T2=model2["T2"])

# initial state matrices
sm1 = statematrix.StateMatrix(density=model1["f"])
sm2 = statematrix.StateMatrix(density=model2["f"])

# saturation effect in mt (pool #2)
sat = operators.R(rL=[0, trf * W])

#
# SPGR
print("SPGR simulation")

rfs = [operators.T(FA, [i * (i + 1) / 2 * PH]) for i in range(Nrf)]
rfs_mt = [operators.T([FA, 0], rf.phi) @ sat for rf in rfs]
seq_spgr_st = [[rf, adc, rx, shift] for rf in rfs]
seq_spgr_bm = [[rf, adc, exg, shift] for rf in rfs]
seq_spgr_mt = [[rf, adc, mt, shift] for rf in rfs_mt]

# Simulations
sim_spgr_st = functions.simulate(seq_spgr_st, max_nstate=100)
sim_spgr_bm = functions.simulate(seq_spgr_bm, max_nstate=100, init=sm1)
sim_spgr_mt = functions.simulate(seq_spgr_mt, max_nstate=100, init=sm2)


# analytical solution
def spgr_sol(model, mt=False, **kwargs):
    ncomp = len(model["T1"])
    I = np.eye(ncomp)
    Theta = np.diag([np.cos(FA / 180 * np.pi)] * ncomp)
    LambdaL = -np.diag(1 / np.array(model["T1"])) - model["kmat"]
    ZetaL = exchange.expm(LambdaL * TR)
    if mt:
        Sigma = np.array([np.sin(FA / 180 * np.pi), 0])
    else:
        Sigma = np.array([np.sin(FA / 180 * np.pi)] * ncomp)
    C = 1 / np.array(model["T1"]) * model["f"]
    return (
        Sigma
        @ np.linalg.inv(I - ZetaL @ Theta)
        @ (ZetaL - I)
        @ np.linalg.inv(LambdaL)
        @ C
    )


sol_spgr_st = spgr_sol(model0)
sol_spgr_bm = spgr_sol(model1)
sol_spgr_mt = spgr_sol(model2, mt=True)


# plot
fig, axes = plt.subplots(nrows=2, num="spgr-x-ph117")
plt.sca(axes[0])
rfindex = np.arange(Nrf) + 1
colors = ["royalblue", "orangered", "goldenrod"]
plt.plot(rfindex, np.abs(sim_spgr_st[:, i117]), color=colors[0], label="EPG")
plt.plot(rfindex, np.abs(sim_spgr_mt[:, i117]), color=colors[1], label="EPG-X(MT)")
plt.plot(rfindex, np.abs(sim_spgr_bm[:, i117]), color=colors[2], label="EPG-X(BM)")
plt.xlim(0, Nrf)
plt.scatter(Nrf, sol_spgr_st, color=colors[0], marker="d")
plt.scatter(Nrf, sol_spgr_mt, color=colors[1], marker="d")
plt.scatter(Nrf, sol_spgr_bm, color=colors[2], marker="d")
plt.xlabel("TR number")
plt.ylabel("Signal / $M_0$")
plt.ylim(0, 0.18)
plt.legend()
plt.grid()
plt.title("SPGR approach to steady-state for $\Phi_0$=117")

plt.sca(axes[1])
plt.plot(PH, np.abs(sim_spgr_st[-1, :]), color=colors[0], label="EPG")
plt.plot(PH, np.abs(sim_spgr_mt[-1, :]), color=colors[1], label="EPG-X(MT)")
plt.plot(PH, np.abs(sim_spgr_bm[-1, :]), color=colors[2], label="EPG-X(BM)")
xlim = PH.min(), PH.max()
plt.scatter(xlim, [sol_spgr_st] * 2, color=colors[0], marker="d")
plt.scatter(xlim, [sol_spgr_mt] * 2, color=colors[1], marker="d")
plt.scatter(xlim, [sol_spgr_bm] * 2, color=colors[2], marker="d")
plt.xlim(*xlim)
plt.ylim(0.04, 0.07)
plt.grid()
plt.xlabel("RF spoil phase")
plt.ylabel("Signal / $M_0$")
plt.title("SPGR steady-state dependence on $\Phi_0$")

plt.tight_layout()


#
# bSSFP
print("bSSFT simulation")

Nssfp = 500
Noffres = 101
delta_res = 0.1e-6 * 3 * 42.6e3
offres = 1 / TR * np.linspace(-0.5, 0.5, Noffres)  # kHz
rf1, rf2 = operators.T(FA, 0), operators.T(FA, 180)
rf1_mt, rf2_mt = operators.T([FA, 0], 0) @ sat, operators.T([FA, 0], 180) @ sat
rlx0 = operators.E(TR, model0["T1"], model0["T2"], g=[offres])
exg1 = operators.X(TR, model1["kmat"], T1=model1["T1"], T2=model1["T2"], g=[offres])
exg1d = operators.X(
    TR, model1["kmat"], T1=model1["T1"], T2=model1["T2"], g=[offres, offres + delta_res]
)
exg2 = operators.X(TR, model2["kmat"], T1=model2["T1"], T2=model2["T2"], g=[offres])

seq_bssfp_st = [[rf1, rlx0], [rf2, rlx0]] * (Nssfp // 2) + [[rf1, adc]]
seq_bssfp_bm = [[rf1, exg1], [rf2, exg1]] * (Nssfp // 2) + [[rf1, adc]]
seq_bssfp_bmd = [[rf1, exg1d], [rf2, exg1d]] * (Nssfp // 2) + [[rf1, adc]]
seq_bssfp_mt = [[rf1_mt, exg2], [rf2_mt, exg2]] * (Nssfp // 2) + [[rf1_mt, adc]]

sim_bssfp_st = functions.simulate(seq_bssfp_st)
sim_bssfp_bm = functions.simulate(seq_bssfp_bm, init=sm1)
sim_bssfp_bmd = functions.simulate(seq_bssfp_bmd, init=sm1)
sim_bssfp_mt = functions.simulate(seq_bssfp_mt, init=sm2)


# analytical solution
def bssfp_sol(model, offres=0, mt=False):
    if not np.isscalar(offres):
        return np.array([bssfp_sol(model, o, mt=mt) for o in offres])

    ncomp = len(model["T1"])
    T1 = model["T1"]
    T2 = model["T2"]
    M0 = model["f"]
    kmat = model["kmat"]
    cshift = model.get("cshift", [0] * ncomp)

    j2pi = 2j * np.pi
    I = np.eye(3 * ncomp)
    Lambda = np.diag(
        sum(
            [
                [-1 / t2 - j2pi * g, -1 / t2 + j2pi * g, -1 / t1]
                for t1, t2, g in zip(T1, T2, cshift)
            ],
            start=[],
        )
    )
    Khi = np.kron(-kmat, np.eye(3))
    Omega = j2pi * np.diag(sum([[-offres, offres, 0] for _ in range(ncomp)], start=[]))
    A = Lambda + Khi + Omega
    Zeta = exchange.expm(A * TR)
    C = np.array(sum([[0, 0, 1 / t1 * f] for t1, f in zip(T1, M0)], start=[]))
    Sigma = np.array(sum([[1, 0, 0] for _ in range(ncomp)], start=[]))
    Theta = np.block(
        [[rf1.mat[0] * (i == j) for j in range(ncomp)] for i in range(ncomp)]
    )
    if mt:
        Theta[-3:, -3:] = sat.mat[1]
    Delta = np.diag(
        sum([[-1, -1, 1] for _ in range(ncomp)], start=[])
    )  # 180° rotation about z
    M = (
        Sigma
        @ np.linalg.inv(Delta - Theta @ Zeta)
        @ Theta
        @ (Zeta - I)
        @ np.linalg.inv(A)
        @ C
    )
    return M


sol_bssfp_st = bssfp_sol(model0, offres)
sol_bssfp_bm = bssfp_sol(model1, offres)
sol_bssfp_bmd = bssfp_sol(dict(model1, cshift=[0, delta_res]), offres)
sol_bssfp_mt = bssfp_sol(model2, offres, mt=True)

# plot
fig, axes = plt.subplots(
    nrows=2, ncols=2, sharex=True, sharey=True, num="bssfp", figsize=(6, 5)
)
plt.sca(axes[0, 0])
plt.plot(offres * 1e3, np.abs(sim_bssfp_st[-1]), label="EPG")
plt.plot(offres * 1e3, np.abs(sol_bssfp_st), "--", label="Direct steady-state")
plt.grid()
plt.legend(loc="upper right")
plt.title("Single compartment")
plt.ylabel("Magnetization (a.u)")

plt.sca(axes[0, 1])
plt.plot(offres * 1e3, np.abs(sim_bssfp_mt[-1]), label="EPG (MT)")
plt.plot(offres * 1e3, np.abs(sol_bssfp_mt), "--", label="Direct steady-state")
plt.grid()
plt.legend(loc="upper right")
plt.title("White matter MT model")

plt.sca(axes[1, 0])
plt.plot(offres * 1e3, np.abs(sim_bssfp_bm[-1]), label="EPG (BM)")
plt.plot(offres * 1e3, np.abs(sol_bssfp_bm), "--", label="Direct steady-state")
plt.grid()
plt.legend(loc="lower right")
plt.title("Myelin water exchange")
plt.xlabel("Off-resonance (Hz)")
plt.ylabel("Magnetization (a.u)")

plt.sca(axes[1, 1])
plt.plot(offres * 1e3, np.abs(sim_bssfp_bmd[-1]), label="EPG (BM)")
plt.plot(offres * 1e3, np.abs(sol_bssfp_bmd), "--", label="Direct steady-state")
plt.grid()
plt.legend(loc="lower right")
plt.title("Myelin water exch. ($\delta$b=0.1ppm)")
plt.ylim(0, 0.2)
plt.xlim(-100, 100)
plt.suptitle("bSSFP off-resonance profiles")

plt.tight_layout()

plt.show()
