"""
states = NamedArray()

# add array
states['main'] = init
states['main] # -> init
np.asarray(states) # -> init[np.newaxis]

# add another array in a group
states.insert('T2', value, group='order1')
states['T2'] # -> value 
states.order1 -> value[np.newaxis] # all names from `order1` group

# get/set all names
states[:] = value
states[..., 0] = value
states *= value

# get/set specific arrays
states['main'] = value
states.order1 = value
states['T2'] = value

"""
import numpy as np


class NamedArray:
    _xp = None
    _array = None
    _indices = None
    _groups = None

    def __init__(self, array=None, names=None, groups=None, *, copy=False):
        self._xp = np # tmp
        if array is None:
            self._array = self._xp.array([])
            self._indices = {}
        else:
            self._array = self._xp.asarray(array, copy=copy)
            self._indices = {name: i for i, name in enumerate(names)}
        self._groups = {key: set(value) for key, value in (groups or {}).items()}

    @property
    def names(self):
        return set(self._indices)

    @property
    def groups(self):
        return set(self._groups)

    def __len__(self):
        return len(self._array)
    
    def __iter__(self):
        return iter(self._indices)
    
    def __contains__(self, name):
        return name in self._indices
    
    def __repr__(self):
        prefix = type(self).__name__ + '('
        suffix = f', names={self.names}, groups={set(self._groups)})'
        arrstr = self._xp.array2string(self._array, prefix=prefix, suffix=suffix)
        return prefix + arrstr + suffix
    
    def __array__(self):
        return self._array
    
    def __getitem__(self, item):
        try:
            indices = self._map(item)
            if isinstance(indices, int):
                return self._array[indices]
            # wrap
            names = set(item)
            groups = {gp: names & _names for gp, _names in self._groups.items()}
            return type(self)(self._array[indices], names, groups)
        except (KeyError, TypeError):
            return self._array[item]
    
    def __setitem__(self, item, value):
        try:
            self._array[self._map(item)] = value
        except (KeyError, TypeError):
            self._array[item] = value
    
    def __getattr__(self, attr):
        try:
            return getattr(self._array, attr)
        except AttributeError:
            pass
        try:
            return self[self._groups[attr]]
        except KeyError:
            raise AttributeError

    def __setattr__(self, attr, value):
        try:
            self[self._groups[attr]] = value
        except (KeyError, AttributeError, TypeError):
            if not attr in dir(type(self)):
                raise AttributeError(f'Cannot set attribute: {attr}')
            super().__setattr__(attr, value)

    def insert(self, name, value, group=None):
        value = self._xp.asarray(value)
        if isinstance(name, str):
            names, values = [name], value[np.newaxis]
        else:
            names, values = list(name), value
        # check names
        duplicates = set(names) & set(self._indices)
        if duplicates:
            raise ValueError(f'Name(s) already exist(s): {duplicates}')
        # store values
        newsize = len(self) + len(names)
        indices = range(len(self), newsize)
        self._indices.update(dict(zip(names, indices)))
        if not len(self):
            self._array = values.copy()
        else:
            self._array.resize((newsize,) + self._array.shape[1:], refcheck=False)
            self._array[indices] = values
        # group
        if group:
            self._groups.setdefault(group, set()).add(name)

    def _map(self, name):
        if isinstance(name, str):
            return self._indices[name]
        return [self._indices[_name] for _name in name]

    def _wrap(self, array=None, *, copy=False):
        """ wrap array """
        array = self._array if array is None else array
        return type(self)(array, self._indices, self._groups, copy=copy)
    
    def _apply(self, ufunc, other, init, inplace=False):
        out = self._array if inplace else None
        if not isinstance(other, NamedArray):
            return self._wrap(ufunc(self._array, other, out=out))
        # merge 
        inter = set(self._indices) & set(other._indices)
        if len(inter) == len(self):
            # same indices
            return self._wrap(ufunc(self._array, other[inter], out=out))
        # different indices
        new = self._wrap(copy=not inplace)
        diff = set(other._indices) - set(new._indices)
        if diff:
            new.add(diff, init)
        ufunc.at(new._array, new._map(inter), other[inter])
        return new
    
    def __add__(self, other):
        return self._apply(self._xp.add, other, 0)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __iadd__(self, other):
        return self._apply(self._xp.add, other, 0, inplace=True)

    def __sub__(self, other):
        return self._apply(self._xp.subtract, other, 0)
    
    def __mul__(self, other):
        return self._apply(self._xp.multiply, other, 0)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __imul__(self, other):
        return self._apply(self._xp.multiply, other, 0, inplace=True)
    
    def __neg__(self):
        return self._wrap(-self._array)
    

if __name__ == '__main__':
    arr1 = NamedArray()
    assert arr1.shape == (0,)
    assert arr1.names == set()

    arr1.insert('A', [[1, 1, 0]])
    assert arr1.shape == (1, 1, 3)
    assert arr1.names == {'A'}
    assert np.all(arr1[:] == [[[1, 1, 0]]])
    assert np.all(arr1['A'] == [[1, 1, 0]])

    arr1[..., -1] = 1
    assert np.all(arr1[:] == [[[1, 1, 1]]])
    arr1 *= -1
    assert np.all(arr1[:] == [[[-1, -1, -1]]])

    arr1.insert('B', [[[1, 1, 0]]], group='G')
    assert arr1.names == {'A', 'B'}
    assert arr1.shape == (2, 1, 3)
    assert np.all(arr1[:] == [[[-1, -1, -1]], [[1, 1, 0]]])
    assert np.all(arr1['B'] == [[1, 1, 0]])

    assert np.all((arr1 + 1)[:] == [[[0, 0, 0]], [[2, 2, 1]]])
    arr1 += 1
    assert np.all(arr1[:] == [[[0, 0, 0]], [[2, 2, 1]]])
    arr1['B'] -= 1
    assert np.all(arr1[:] == [[[0, 0, 0]], [[1, 1, 0]]])

    arr2 = arr1[['B']]
    assert arr2.names == {'B'}
    assert np.all(arr2[:] == [[[1, 1, 0]]])
    assert np.all((arr1 - arr2)[:] == [[[0, 0, 0]], [[0, 0, 0]]])

    arr1.G += 3
    assert np.all(arr1[:] == [[[0, 0, 0]], [[4, 4, 3]]])

    arr1.G += arr2.G
    assert np.all(arr1[:] == [[[0, 0, 0]], [[5, 5, 3]]])
