# Week 1

from math import log2
import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress

def infocontent(p):
    return - log2(p)

def entropy(p: np.array):
    if type(p) == list:
        p = np.array(p)
    if np.sum(p) != 1:
        raise Exception('The sum of the elements of p should be = 1: {}'.format(p))
    with suppress(ZeroDivisionError):
        H = - np.dot(p, np.where(p > 0, np.log2(p), 0))

    return  H


