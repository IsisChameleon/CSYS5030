# Week 1

from math import log2, isclose
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
    # if np.sum(p) != 1:
    if isclose(np.sum(p), 1, rel_tol=1e-6):
        raise Exception('The sum of the elements of p should be = 1: {}'.format(p))
    with suppress(ZeroDivisionError):
        H = - np.dot(p, np.where(p > 0, np.log2(p), 0))

    return  H

# Calculate entropy of a sequence of stuff

def calculateDatasetShannonEntropy(items):
    # also called empiricalentropy in course
    itemscount = collections.Counter(items)
    
    # probability = # item x / # total number of items
    dist = [x/sum(itemscount.values()) for x in itemscount.values()]
 
    # use scipy to calculate entropy
    entropy_value = scips.entropy(dist, base=2)
 
    return entropy_value

entropyempirical = lambda items : calculateDatasetShannonEntropy(items)

def jointEntropy(p: np.array):

    if type(p) == list:
        p = np.array(p)

    if not isclose(np.sum(p), 1, rel_tol=1e-6):
        raise Exception('The sum of the elements of p should be = 1{}'.format(p))

    # Perform element-wise p(x,y)*log p(x,y)
    plogp = lambda x: - x * np.where(x > 0, np.log2(x), 0)
    Hs = np.array([plogp(pij) for pij in p])

    return  np.sum(Hs)

def marginalP(p: np.array, dim: int):
    if dim + 1 > p.ndim:
        raise Exception('The probability matrix has only {} dimensions while you are requesting to get marginal in dimension {}'.format(p.ndim, dim+1))
    return np.sum(p, axis=dim)

marginalX = lambda  p : marginalP(p, 0)

marginalY = lambda  p : marginalP(p, 1)

def conditionalentropy2(p: np.array):
    # entropyXGivenY
    # H(X|Y)
    # 2 dim only for now

    # e.g.
    # p(x,y)
    # coded for Y rows (dim=0) \X   columns (dim=1)
    # =>
    # transpose(p)
    # p = np.array([[0.125, 0.0625, 0.03125, 0.03125],
    #       [0.0625, 0.125, 0.03125, 0.03125],
    #       [0.0625, 0.0625, 0.0625, 0.0625],
    #       [0.25,0,0,0]])
    # Correct answer = 11/8 = 1.375 fro transpose p                    
    if (type(p) == list):
        p = np.array(p)

    p = np.transpose(p)

    total = 0
    for i in range(p.shape[0]):
        pyi = np.sum(p[i])
        total+= pyi * entropy1(1/pyi * p[i])

    return total

def conditionalentropy(p: np.array):
    # entropyXGivenY
    # H(X|Y)
    # 2 dim only for now
    # X = rows
    # Y = columns
                  
    if (type(p) == list):
        p = np.array(p)

    total = 0
    for i in range(p.shape[1]):
        pxi = np.sum(p[:,i])
        total+= pxi * entropy1(1/pxi * p[:,i])

    return total

