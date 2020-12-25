import numpy as np


def get_simplex_shapes(shapes):
    return [np.prod(shape) for shape in shapes.values()]
