import time
import epgpy as epg
from epgpy import parallel


def dummy(sm, **kwargs):
    return sm.shape

if __name__ == '__main__':
    nOP = 1000000
    sm0 = epg.StateMatrix([1,1,1])
    rlx = epg.R(1, 1, r0=1)

    func = rlx.derive0
    # func = dummy

    print(f'Start apply (nOP={nOP})')
    tic = time.time()
    res = parallel.apply(func, {i: sm0 for i in range(nOP)}, single=True, inplace=False, how='single')
    # res = parallel.apply(func, {i: sm0 for i in range(nOP)}, single=True, inplace=False, how='multiprocessing')
    print(f'Done ({time.time() - tic:.0f}s)')

    # print(res)

