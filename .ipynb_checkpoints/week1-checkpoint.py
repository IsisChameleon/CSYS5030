# Week 1

from math import log2
import numpy as np

def infocontent(p):
    return - log2(p)

def entropy(p):
    return  - np.dot(p, log2(p))