import pytest
from epgpy import utils


def test_axes_class():
    axes = utils.Axes("ax1", "ax2")
    arr = [1, 2]
    assert arr[axes.ax1] == 1
    assert arr[axes.ax2] == 2
    assert len(axes) == 2
    assert axes["ax1"] == 0
    assert axes["ax2"] == 1
