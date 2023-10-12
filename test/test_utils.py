import numpy as np
from epgpy import utils


def test_axes_class():
    axes = utils.Axes("ax1", "ax2")
    arr = [1, 2]
    assert arr[axes.ax1] == 1
    assert arr[axes.ax2] == 2
    assert len(axes) == 2
    assert axes["ax1"] == 0
    assert axes["ax2"] == 1


def test_lazygetter():
    class Obj:
        a = "foo"

        @property
        def b(self):
            return np.arange(len(self.a))

    obj = Obj()

    getter = utils.DeferredGetter(obj, ["a", "b"])
    # setup obj
    assert getter["a"] == "foo"
    assert np.all(getter["b"] == np.arange(3))
    obj.a = "foobar"
    assert getter["a"] == "foobar"
    assert np.all(getter["b"] == np.arange(6))
