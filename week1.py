# Week 1

from math import log2, isclose
import numpy as np
import matplotlib.pyplot as plt
from contextlib import suppress
import collections
import scipy.stats as scips

def marginalP(p: np.array, dim: int):
    if dim + 1 > p.ndim:
        raise Exception('The probability matrix has only {} dimensions while you are requesting to get marginal in dimension {}'.format(p.ndim, dim+1))
    return np.sum(p, axis=dim)

marginalX = lambda  p : marginalP(p, 0)

marginalY = lambda  p : marginalP(p, 1)

def infocontent(p):
    return - log2(p)

def entropy(p: np.array):

    if type(p) == list:
        p = np.array(p)

    # if np.sum(p) != 1:
    if not isclose(np.sum(p), 1, rel_tol=1e-6):
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
Takes a array of 2D joint probabilities 
returns the mutual information I(X;Y)

I(X;Y)=H(X)+H(Y)−H(X,Y)

Tests:
mutualinformation([0.2, 0.3; 0.1, 0.4]) and validating that you get the result 0.0349 bits
mutualinformation([0.5, 0; 0, 0.5]) and validating that you get the result 1 bit
mutualinformation([0.25, 0.25; 0.25, 0.25]) and validating that you get the result 0 bits


'''
def mutualInformation(p: np.array):

    if type(p) == list:
        p = np.array(p)

    if not isclose(np.sum(p), 1, rel_tol=1e-6):
        raise Exception('The sum of the elements of p should be = 1{}'.format(p))

    Hx = entropy(marginalX(p))
    Hy = entropy(marginalY(p))
    Hxy = jointEntropy(p)

    return Hx + Hy - Hxy

'''
Takes a array of 2D samples as input
returns the joint entropy

Tests:
entropyempirical(['a','b','c','d']) should return 2
testdata2bits = [[0,0],[0,1],[1,0],[1,1]] should return 2

'''

def jointEntropyEmpiricalOld(samples: np.array):

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

#----------------------------------------------------------
#  JOINT ENTROPY EMPIRICAL
#----------------------------------------------------------

def jointEntropyEmpirical(samples: np.array):

    # samples : 
    # each row represent a sample
    # columns represent the features or random variables of interest

    # Same as above but when more than 2 dimensions

    if type(samples) == list:
        samples=np.array(samples)

    if samples.ndim == 1:
        samples = samples.reshape(samples.shape[0], 1)

    N, D = samples.shape   

    alphabets=list()
    for d in range(D):
        #print('set samples{} = {}'.format(d, set(samples[:,d])))
        alphabets.append(list(set(samples[:,d])))

    alphabetsDimensions = tuple([ len(alphabeti) for alphabeti in alphabets])

    jointProbabilities=np.zeros(alphabetsDimensions)

    for i in range(samples.shape[0]):
        #print('Sample {} : {}'.format(i, samples[i]))
        sample=samples[i]
        sampleFeaturesIndex = tuple([alphabets[d].index(sample[d]) for d in range(D)])
        jointProbabilities[sampleFeaturesIndex]+=1

    jointProbabilities/=N

    return jointEntropy(jointProbabilities)

'''
Mutual Information Empirical using data
samples : 
    # each row represent a sample xi, yi, ...
    # columns represent the features or random variables of interest

    I(X;Y)=H(X)+H(Y)−H(X,Y)

    mutualinformationempirical([0,0,1,1],[0,1,0,1]) and validating that you get the result 0 bits
    mutualinformationempirical([0,0,1,1],[0,0,1,1]) and validating that you get the result 1 bit
'''

def mutualInformationEmpiricalOld(samples: np.array):

    if type(samples) == list:
        samples=np.array(samples)

    Hxy = jointEntropyEmpirical(samples)
    Hx = entropyEmpirical(samples[:,0])
    Hy = entropyEmpirical(samples[:,1])
    return Hx + Hy - Hxy

#----------------------------------------------------------
#  MUTUAL INFORMATION EMPIRICAL
#----------------------------------------------------------

def mutualInformationEmpirical(samples: np.array):

    if type(samples) == list:
        samples=np.array(samples)

    Hxy = jointEntropyEmpirical(samples)
    Hx = jointEntropyEmpirical(samples[:,0])
    Hy = jointEntropyEmpirical(samples[:,1])
    return Hx + Hy - Hxy

#----------------------------------------------------------
#  MUTUAL INFORMATION EMPIRICAL Xn , Yn
#----------------------------------------------------------

def mutualInformationEmpiricalXnYn(xn, yn):

    if type(xn) == list:
        xn=np.array(xn)

    if type(yn) == list:
        yn=np.array(yn)

    if (xn.shape[0] != yn.shape[0]):
        raise Exception('Xn and Yn should have the same number of rows/samples {} vs {}'.format(xn.shape[0], yn.shape[0]))

    if xn.ndim == 1:
        xn=xn.reshape(xn.shape[0],1)

    if yn.ndim == 1:
        yn=yn.reshape(yn.shape[0],1)

    Hxy = jointEntropyEmpirical(np.column_stack((xn,yn)))
    Hx = jointEntropyEmpirical(xn)
    Hy = jointEntropyEmpirical(yn)
    return Hx + Hy - Hxy


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

#----------------------------------------------------------
#  CONDITIONAL ENTROPY EMPIRICAL XN YN
#----------------------------------------------------------
'''
 CONDITIONAL ENTROPY EMPIRICAL USING XN AND YN (YN CAN BE MULTIVARIATE)

 TESTS:
 conditionalentropyempirical([0,0,1,1],[0,1,0,1]) and validating that you get the result 1 bit.
 conditionalentropyempirical([0,0,1,1],[0,0,1,1]) and validating that you get the result 0 bits.
'''
def conditionalEntropyEmpiricalXnYn(xn, yn):
    # samples : 
    # each row represent a sample
    # columns represent the features or random variables of interest

    if type(xn) == list:
        xn=np.array(xn)

    if type(yn) == list:
        yn=np.array(yn)

    if (xn.shape[0] != yn.shape[0]):
        raise Exception('Xn and Yn should have the same number of rows/smaples {} vs {}'.format(xn.shape[0], yn.shape[0]))

    if xn.ndim == 1:
        xn=xn.reshape(xn.shape[0],1)

    if yn.ndim == 1:
        yn=yn.reshape(yn.shape[0],1)

    # We need to compute H(X,Y) - H(X):

    H_XY = jointEntropyEmpirical(np.column_stack((xn,yn)))
    H_Y = jointEntropyEmpirical(yn)
    return H_XY - H_Y

#-------------------------------------------------------------------------------------------------------------------------------
#
# CONDITIONAL MUTUAL INFORMATION EMPIRICAL 
#
# The conditional mutual information between variables x and y, conditional on variable z, for a distribution p(x,y,z) is:
# 
# I(X;Y∣Z)=H(X∣Z)+H(Y∣Z)−H(X,Y∣Z)
#------------------------------------------------------------------------------------------------------------------------------

'''
Test that your code works by running, e.g.:
conditionalmutualinformationempirical([0,0,1,1],[0,1,0,1],[0,1,0,1]) and validating that you get the result 0 bits.
conditionalmutualinformationempirical([0,0,1,1],[0,0,1,1],[0,1,1,0]) and validating that you get the result 1 bit.
conditionalmutualinformationempirical([0,0,1,1],[0,1,0,1],[0,1,1,0]) and validating that you get the result 1 bit. 
'''

def conditionalMutualInformationEmpirical(xn: np.array, yn: np.array, zn: np.array):
    
    HXgZ = conditionalEntropyEmpirical([[x, z] for x,z in zip(xn, zn)])
    HYgZ = conditionalEntropyEmpirical([[y, z] for y,z in zip(yn, zn)])
    xnyn = np.column_stack((xn,yn))
    HXYgZ = conditionalEntropyEmpiricalXnYn(xnyn, zn)
    CMI = HXgZ + HYgZ - HXYgZ
    return CMI

