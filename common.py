import numpy as np

__cross_matrix__ = np.array([[[0, 0, 0],
                              [0, 0, -1.0],
                              [0, 1.0, 0]],
                             [[0, 0, 1.0],
                              [0, 0, 0],
                              [-1.0, 0, 0]],
                             [[0, -1.0, 0],
                              [1, 0, 0],
                              [0, 0, 0]]])


def norm(v, axis=None):
    return np.sqrt(np.sum(v * v, axis=axis))


def skew(v):
    return __cross_matrix__.dot(v).squeeze()