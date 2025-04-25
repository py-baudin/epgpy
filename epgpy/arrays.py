"""

states = SectionArray('_', init, ['section1', 'section2'])

# main section
states['_'] # -> init
np.asarray(states) # -> init[np.newaxis]

# add name
states.add('group1', 'first', value)
states['first'] # -> value 
states.group1 -> value[np.newaxis] # all names from group1

# get/set all names
states[:] = value
states[..., 0] = value
states *= value

# get/set specific names
states['_'][:] = value
states.group1[:] = value
states[''first'][:] = value

"""

import functools
from collections import namedtuple
import numpy as np


# Section 
Item = namedtuple('Item', ['section', 'name'])

class SectionArray:
    
    def __init__(self, name, init, sections):
        self.array = np.asarray(init)[np.newaxis]
        self._items = [Item("", name)]
        self._names = {name: 0}
        self._sections = {section: slice(1,1) for section in sections}

    def add(self, section, name, array):
        array = np.asarray(array)
        assert array.shape == self.array.shape[1:]
        if not section in self._sections:
            raise ValueError(f'Section "{section}" doest not exist')
        if name in self._names:
            raise ValueError(f'Name: "{section}:{name}" already exists')
        # sort items
        self._items = sorted(self._items + [Item(section, name)])
        # reset indices and slices
        first, num = 1, 0
        for i, item in enumerate(self._items[1:]):
            self._names[item.name] = i + 1
            if item.section != self._items[first].section:
                first, num = i + 1, 0
            num += 1
            self._sections[item.section] = slice(first, first + num)
        # store
        index = self._names[name]
        self.array = np.r_[self.array[:index], array[np.newaxis], self.array[index:]]

    def new(self, array=None):
        """ array wrapper"""
        array = self.array if array is None else np.asarray(array)
        assert array.shape == self.array.shape
        obj = self.__new__(type(self))
        obj.array = array
        obj._items = self._items
        obj._names = self._names
        obj._sections = self._sections
        return obj
          
    def __repr__(self):
        return repr(self.array) + f'(sections: {list(self._sections)}, names: {list(self._names)}'

    def __array__(self):
        return self.array
    
    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            return self.array[self._names[item]]
        return self.array[item]
    
    def __setitem__(self, item, value):
        if isinstance(item, str):
            self.array[self._names[item]] = value
            return
        self.array[item] = value
    
    def __getattr__(self, attr):
        try:
            return getattr(self.array, attr)
        except AttributeError:
            pass
        try:
            return self.array[self._sections[attr]]
        except KeyError:
            raise AttributeError

    # numeric and comparison operators 

    OPERATORS = (
        '__add__', '__iadd__', '__radd__',
        '__sub__', '__isub__', '__rsub__',
        '__mul__', '__imul__', '__rmul__',
        '__pow__', '__ipow__', '__rpow__',
        # '__matmul__', '__imatmul__', '__rmatmul__',
        '__truediv__', '__itruediv__', '__rtruediv__',
        '__floordiv__', '__ifloordiv__', '__rfloordiv__',
        '__mod__', '__imod__', '__rmod__',
        '__divmod__', '__rdivmod__',
        '__eq__', '__ne__', '__ge__', '__gt__', '__le__', '__lt__',
        '__neg__',
    )

    def _setop(locals_, op):
        def wrapper(self, *args):
            return self.new(getattr(self.array, op)(*args))
        wrapper.__name__ = op
        wrapper.__qualname__ = op
        locals_[op] = wrapper
    
    for op in OPERATORS:
        _setop(locals(), op)

    del _setop, OPERATORS

if __name__ == '__main__':
    states = SectionArray('main', [[[0, 0, 0]]], ['A', 'B'])
    states.add('A', 'A2', [[[1, 1, 0]]])
    states[..., -1] = 1
    states *= -1

    states.add('A', 'A1', [[[1, 1, 0]]])
    states.add('B', 'B1', [[[2, 2, 0]]])

