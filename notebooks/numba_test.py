import numba
from numba import jit
import time
import numpy as np

foo = np.random.rand(10000)
bar = np.random.rand(10000)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def dot(a: np.ndarray, b: np.ndarray):
    sum = 0.
    for i in range(a.shape[0]):
        sum += a[i] * b[i]
    return sum


def slow_dot(a: np.ndarray, b: np.ndarray):
    sum = 0.
    for i in range(a.shape[0]):
        sum += a[i] * b[i]
    return sum


def np_dot(a: np.ndarray, b: np.ndarray):
    return a @ b