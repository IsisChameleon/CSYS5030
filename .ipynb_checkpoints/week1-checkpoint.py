# Week 1

from math import log2
import numpy as np

def infocontent(p):
    return - log2(p)

def entropy(p: np.array):
    if np.sum(p) != 1:
        raise Exception('The sum of the elements of p should be = 1: {}'.format(p))
    return  - np.dot(p, np.log2(p))

