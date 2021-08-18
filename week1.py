# Week 1

from math import log2
import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress
import collections
import scipy.stats as scips

def infocontent(p):
    return - log2(p)

def entropy1(p: np.array):
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

def jointEntropy(p: np.array):
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

def marginalP(p: np.array, dim: int):
    if dim + 1 > p.ndim:
        raise Exception('The probability matrix has only {} dimensions while you are requesting to get marginal in dimension {}'.format(p.ndim, dim+1))
    return np.sum(p, axis=dim)

marginalX = lambda  p : marginalP(p, 0)

marginalY = lambda  p : marginalP(p, 1)

def entropyXGivenY(p: np.array):
    # 2 dim only for now

    # e.g.
    # p(x,y)
    # Y rows (dim=0) \X   columns (dim=1)
    # p = np.array([[0.125, 0.0625, 0.03125, 0.03125],
    #       [0.0625, 0.125, 0.03125, 0.03125],
    #       [0.0625, 0.0625, 0.0625, 0.0625],
    #       [0.25,0,0,0]])
    # Correct answer = 11/8 = 1.375
    total = 0
    for i in range(p.shape[0]):
        pyi = np.sum(p[i])
        total+= pyi * entropy1(1/pyi * p[i])

    return total
