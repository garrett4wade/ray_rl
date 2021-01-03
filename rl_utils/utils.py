import numpy as np
from sys import getsizeof, stderr
from itertools import chain
from collections import deque


class StorageProperty:
    """defines basic properties for a specific shared storage block

    Hopefully, all data needed can be concatenated along the last dim, such that
    communication/copy from Ray to main process has the lowest cost.

    However, there are different types of data that can not concatenated together
    e.g. centralized & decentralized data in multiagent environments,
         burn_in data & data for backpropagation
    (or we MAY concatenate them together at the cost of memory)

    Hence we classify all data we need into different storage types, and
    concatenate data of the same type to reduce communication overhead.
    """
    def __init__(self, length, agent_num, keys, simplex_shapes):
        self.length = length
        self.keys = keys
        self.agent_num = agent_num
        self.simplex_shapes = simplex_shapes
        self.split = [sum(self.simplex_shapes[:i]) for i in range(1, len(self.simplex_shapes))]
        self.combined_shape = (sum(self.simplex_shapes), ) if agent_num == 1 else (agent_num, sum(self.simplex_shapes))


def get_simplex_shapes(shapes):
    return [int(np.prod(shape)) for shape in shapes.values()]


def get_total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Reference: https://code.activestate.com/recipes/577504/

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: lambda d: chain.from_iterable(d.items()),
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)
        if verbose:
            print(s, type(o), repr(o), file=stderr)
        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s
    return sizeof(o)
