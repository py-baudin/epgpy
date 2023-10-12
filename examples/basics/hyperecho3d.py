import numpy as np
from matplotlib import pyplot as plt

from epgpy import operators, plotting


nrf = 3
exc = operators.T(90, 90)
rfc = operators.T(180, 0)
rf1 = operators.T(30, 0)
rf2 = operators.T(-30, 0)

# integer gradients (arbitrary units)
ks = np.random.randint(1, 5, (5, 3))
ks *= 2 * np.random.randint(0, 2, (5, 3)) - 1  # random sign

# float gradient values (rad/m)
ks = np.random.uniform(-5, 5, (5, 3))

# 1d gradients
gs1 = [operators.S(k[0], duration=1) for k in ks]
gs2 = [operators.S(k[0], duration=1) for k in ks[::-1]]
# hyper-echo sequence
seq = [[rf1, g] for g in gs1] + [rfc] + [[g, rf2] for g in gs2]

plotting.plot_epg(
    seq,
    kgrid=0.01,
    title="Hyper-echo sequence (random 1d gradients)",
    figname="hyperecho-1d",
)


# 2d gradients
gs1 = [operators.S(k[:2], duration=1) for k in ks]
gs2 = [operators.S(k[:2], duration=1) for k in ks[::-1]]
# hyper-echo sequence
seq = [[rf1, g] for g in gs1] + [rfc] + [[g, rf2] for g in gs2]

plotting.plot_epg(
    seq,
    kgrid=0.01,
    title="Hyper-echo sequence (random 2d gradients)",
    figname="hyperecho-2d",
)

# 3d gradients
gs1 = [operators.S(k, duration=1) for k in ks]
gs2 = [operators.S(k, duration=1) for k in ks[::-1]]
# hyper-echo sequence
seq = [[rf1, g] for g in gs1] + [rfc] + [[g, rf2] for g in gs2]

plotting.plot_epg(
    seq,
    kgrid=0.01,
    title="Hyper-echo sequence (random 3d gradients)",
    figname="hyperecho-3d",
)

plotting.show()
