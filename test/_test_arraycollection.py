import numpy as np
import pytest
from epgpy import arraycollection


def test_arraycollection_class():
    coll = arraycollection.ArrayCollection()
    a = np.arange(1 * 4).reshape(1, 4)
    b = np.arange(3 * 1).reshape(3, 1)

    coll.set("a", a)
    assert coll.shape == (1, 4)
    assert coll.ndim == 2
    assert np.allclose(coll.get("a"), a)
    assert not coll.get("a") is a  # copy

    coll.set("b", b)
    assert coll.shape == (3, 4)
    assert coll.get("a").shape == (3, 4)
    assert coll.get("b").shape == (3, 4)
    assert np.allclose(coll.get("a"), a)
    assert np.allclose(coll.get("b"), b)

    # expand
    c = np.arange(2 * 1).reshape(2, 1, 1)
    coll.set("c", c)
    assert coll.shape == (2, 3, 4)
    assert coll.get("a").shape == (2, 3, 4)
    assert coll.get("b").shape == (2, 3, 4)
    assert coll.get("c").shape == (2, 3, 4)

    # pop
    coll.pop("a")
    assert coll.shape == (2, 3, 1)
    coll.pop("c")
    assert coll.shape == (3, 1)

    #
    # kdim and insert_index
    coll = arraycollection.ArrayCollection(kdim=1)
    a = np.arange(4 * 2).reshape(4, 2)
    b = np.arange(3).reshape(3, 1, 1)

    coll.set("a", a)
    assert coll.shape == (4,)

    coll.set("b", b)
    assert coll.shape == (3, 4)

    assert coll.get("a").shape == (3, 4, 2)
    assert coll.get("b").shape == (3, 4, 1)

    #
    # resize
    coll.resize_axis(6, axis=-1)
    assert coll.shape == (3, 6)

    coll.expand(3, insert_index=-1)
    assert coll.shape == (3, 1, 6)

    coll.resize((3, 2, 6))
    assert coll.shape == (3, 2, 6)
