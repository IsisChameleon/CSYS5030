# Week 1

from math import log2, isclose
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

    # if np.sum(p) != 1:
    if not isclose(np.sum(p), 1, rel_tol=1e-1):
        raise Exception('The sum of the elements of p should be = 1: {}'.format(p))

    with np.errstate(divide='ignore'):
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

def entropyEmpirical(items):
    itemscount = collections.Counter(items)
    
    # probability = # item x / # total number of items
    dist = [x/sum(itemscount.values()) for x in itemscount.values()]
 
    # use scipy to calculate entropy
    entropy_value = entropy(dist)
 
    return entropy_value

def getProbability(items):
    itemscount = collections.Counter(items)
    
    # probability = # item x / # total number of items
    dist = [x/sum(itemscount.values()) for x in itemscount.values()]

    return dist



entropyempirical = lambda items : calculateDatasetShannonEntropy(items)

def jointEntropy(p: np.array):

    if type(p) == list:
        p = np.array(p)

    if not isclose(np.sum(p), 1, rel_tol=1e-6):
        raise Exception('The sum of the elements of p should be = 1{}'.format(p))

    # Perform element-wise p(x,y)*log p(x,y)
    plogp = lambda x: - x * np.where(x > 0, np.log2(x), 0)
    with np.errstate(divide='ignore'):
        Hs = np.array([plogp(pij) for pij in p])

    return  np.sum(Hs)

'''
Takes a array of 2D samples as input
returns the joint entropy

Tests:
entropyempirical(['a','b','c','d']) should return 2
testdata2bits = [[0,0],[0,1],[1,0],[1,1]] should return 2

'''

def jointEntropyEmpirical(samples: np.array):

    # samples : 
    # each row represent a sample
    # columns represent the features or random variables of interest

    if type(samples) == list:
        samples=np.array(samples)

    N, D = samples.shape

    if D > 2:
        raise NotImplementedError

    alphabetX=list(set(samples[:,0]))
    alphabetY=list(set(samples[:,1]))

    jointProbabilities=np.zeros((len(alphabetX), len(alphabetY)))

    for i in range(samples.shape[0]):
        #print('Sample {} : {}'.format(i, samples[i]))
        sample=samples[i]
        jointProbabilities[alphabetX.index(sample[0]), alphabetY.index(sample[1])]+=1

    jointProbabilities/=N

    return jointEntropy(jointProbabilities)

'''

'''
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
        total+= pyi * entropy(1/pyi * p[i])

    return total

def conditionalEntropy(p: np.array):
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
        total+= pxi * entropy(1/pxi * p[:,i])

    return total

def conditionalEntropyEmpirical(samples):
    # samples : 
    # each row represent a sample
    # columns represent the features or random variables of interest

    if type(samples) == list:
        samples=np.array(samples)

    N, D = samples.shape

    if D > 2:
        raise NotImplementedError

    jointEntropy = jointEntropyEmpirical(samples)
    entropyY = entropyEmpirical(samples[:,1].flatten())
    return jointEntropy - entropyY

