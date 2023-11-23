import numpy as np


def uniform(x, lower_bound, upper_bound):
    if lower_bound < x < upper_bound:
        return 0.0
    return -np.inf


def inverse_bounded(x, lower_bound, upper_bound):
    if lower_bound < x < upper_bound:
        return -np.log(x)
    return -np.inf


def chi_square(x, center, e_scale, lower_bound=0.0, upper_bound=None):
    if x < lower_bound:
        return -np.inf
    if upper_bound != None:
        if x > upper_bound:
            return -np.inf
    return -0.5*((x-center)/e_scale)**2
