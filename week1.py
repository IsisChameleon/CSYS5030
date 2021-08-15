# Week 1

from math import log2
import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress
import collections
import scipy.stats as scips

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

# Calculate entropy of a sequence of stuff

 
def calculateDatasetShannonEntropy(items):
    itemscount = collections.Counter(items)
    
    # probability = # item x / # total number of items
    dist = [x/sum(itemscount.values()) for x in itemscount.values()]
 
    # use scipy to calculate entropy
    entropy_value = scips.entropy(dist, base=2)
 
    return entropy_value

def jointentropy(p: np.array):
    print(type(p))
    print(p.shape[0])
    print(p.shape[1])
    if type(p) == list:
        p = np.array(p)
    if p.shape[0] != p.shape[1]:
        raise Exception("p must be a square matrix")
    print("np.sum(p, axis=0) != 1 :", np.sum(p, axis=0) != 1)
    print("np.sum(p, axis=0):", np.sum(p, axis=0))
    if ( np.sum(p, axis=0) != 1).any() or (np.sum(p, axis=1) != 1).any():
        raise Exception('The sum of the elements of p should be = 1 in all dimensions {}'.format(p))
    with suppress(ZeroDivisionError):
        H = - np.dot(p, np.where(p > 0, np.log2(p), 0))
        H = np.sum(H)

    return  H