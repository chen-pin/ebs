import numpy as np


def uniform(x, lower_bound, upper_bound):
    if lower_bound < x < upper_bound:
        return 0.0
    return -np.inf


def inverse_bounded(x, lower_bound, upper_bound):
    if lower_bound < x < upper_bound:
        return -np.log(x)
    return -np.inf
