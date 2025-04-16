import numpy as np
from epgpy import common, functions
from epgpy import operators as ops, statematrix
import pytest

try:
    import cupy
except ImportError:
    cupy = None


def test_map_arrays():
    map_arrays = common.map_arrays
    arrs = map_arrays([None, 1, [1, 2]])
    assert arrs[:2] == [None, 1]
    assert isinstance(arrs[2], np.ndarray)
    assert np.all(arrs[2] == [1, 2])

    arrs = map_arrays({"a": None, "b": 1, "c": [1, 2]})
    assert arrs["a"] is None
    assert arrs["b"] == 1
    assert isinstance(arrs["c"], np.ndarray)
    assert np.all(arrs["c"] == [1, 2])

    arrs = map_arrays(a=None, b=1, c=[1, 2])
    assert arrs["a"] is None
    assert arrs["b"] == 1
    assert isinstance(arrs["c"], np.ndarray)
    assert np.all(arrs["c"] == [1, 2])

    arrs = map_arrays(
        {"a": None, "b": 1, "c": [1, 2]}, func=lambda arr: np.expand_dims(arr, (0, 2))
    )
    assert arrs["a"] is None
    assert arrs["b"] == 1
    assert isinstance(arrs["c"], np.ndarray)
    assert arrs["c"].shape == (1, 2, 1)


def test_broadcastable():
    assert common.broadcastable((1,), (1,))
    assert common.broadcastable((3,), (1,))
    assert common.broadcastable((3, 1), (1, 2))
    assert common.broadcastable((2,), (1, 2))

    # prepend dimensions
    assert common.broadcastable((2,), (3, 2))
    assert not common.broadcastable((3,), (3, 2))

    # append dimensions
    assert common.broadcastable((3,), (3, 2), append=True)
    assert not common.broadcastable((2,), (3, 2), append=True)


def test_broadcast_shapes():
    assert common.broadcast_shapes((1,)) == (1,)
    assert common.broadcast_shapes((1, 3), (2, 1)) == (2, 3)
    assert common.broadcast_shapes((3,), (2, 3)) == (2, 3)
    with pytest.raises(ValueError):
        assert common.broadcast_shapes((2,), (2, 3)) == (2, 3)

    assert common.broadcast_shapes((2,), (2, 3), append=True) == (2, 3)
    with pytest.raises(ValueError):
        assert common.broadcast_shapes((3,), (2, 3), append=True) == (2, 3)


def test_expand_objects():
    assert common.expand_arrays(None)[0] is None
    assert common.expand_arrays(1.2)[0] == 1.2
    assert np.allclose(common.expand_arrays([1])[0], [1])
    assert np.allclose(common.expand_arrays([[1, 2]])[0], [[1, 2]])

    # expand
    a1, a2, a3 = common.expand_arrays(None, [[1], [2]], [3, 4, 5])
    assert a1 is None
    assert a2.shape == (2, 1)
    assert a3.shape == (1, 3)
    assert np.allclose(a2, [[1], [2]])
    assert np.allclose(a3, [[3, 4, 5]])

    with pytest.raises(ValueError):
        common.expand_arrays(None, [1, 2], [[3, 4, 5]])

    a1, a2, a3 = common.expand_arrays(None, [1, 2], [[3, 4, 5]], append=True)
    assert a1 is None
    assert a2.shape == (2, 1)
    assert a3.shape == (1, 3)
    assert np.allclose(a2, [[1], [2]])
    assert np.allclose(a3, [[3, 4, 5]])

    with pytest.raises(ValueError):
        common.expand_arrays(None, [[1], [2]], [3, 4, 5], append=True)


@pytest.mark.skipif(cupy is None, reason="cupy not found")
def test_array_module_cupy():
    """test alternative array modules"""
    import cupy as cp

    # test is_array_module and get_array_module
    nparr = np.array([0])
    cparr = cp.array([0])

    assert common.get_array_module(nparr) == np
    assert common.get_array_module([nparr]) == np
    assert common.get_array_module([nparr, 1]) == np
    with pytest.raises(ValueError):
        common.get_array_module([nparr, cparr])

    assert common.is_array_module(nparr, np)
    assert common.is_array_module(nparr, "np")
    assert common.is_array_module(nparr, "numpy")
    assert common.is_array_module([nparr], "numpy")
    assert common.is_array_module([nparr, 1], "numpy")
    assert not common.is_array_module(cparr, "numpy")
    assert not common.is_array_module([1], "numpy")
    assert not common.is_array_module("foobar", "numpy")

    assert common.is_array_module(cparr, cp)
    assert common.is_array_module(cparr, "cp")
    assert common.is_array_module(cparr, "cupy")

    #
    # test setting array module to cupy

    # current array module
    xp = common.get_array_module()

    common.set_array_module("cupy")
    assert common.get_array_module() is cp
    exc = ops.T(90, 90)
    ref = ops.T(150, 0)
    shift = ops.S(1)
    relax = ops.E(5, 1e3, [10, 20, 30], g=[[0, 0.1]])
    adc = ops.ADC
    seq = [exc] + [shift, relax, ref, shift, relax, adc] * 10

    smcp = statematrix.StateMatrix()
    for op in seq:
        smcp = op(smcp)
    sigcp = functions.simulate(seq)

    # numpy
    common.set_array_module("numpy")
    assert common.get_array_module() is np
    exc = ops.T(90, 90)
    ref = ops.T(150, 0)
    shift = ops.S(1)
    relax = ops.E(5, 1e3, [10, 20, 30], g=[[0, 0.1]])
    adc = ops.ADC
    seq = [exc] + [shift, relax, ref, shift, relax, adc] * 10

    smnp = statematrix.StateMatrix()
    for op in seq:
        smnp = op(smnp)
    signp = functions.simulate(seq)

    assert np.allclose(smcp.states, smnp.states)
    assert np.allclose(sigcp, signp)

    # revert to current module
    common.set_array_module(xp)


def test_deferredgetter():
    class Obj:
        a = "foo"

        @property
        def b(self):
            return np.arange(len(self.a))

    obj = Obj()

    getter = common.DeferredGetter(obj, ["a", "b"])
    # setup obj
    assert getter["a"] == "foo"
    assert np.all(getter["b"] == np.arange(3))
    obj.a = "foobar"
    assert getter["a"] == "foobar"
    assert np.all(getter["b"] == np.arange(6))
