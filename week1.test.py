import unittest
import numpy as np
import numpy.testing as npt
from numpy.testing._private.utils import assert_array_equal


from week1 import infocontent, entropy, marginalP, marginalX, marginalY, mutualInformation

class TestEntropy(unittest.TestCase):

    def test_informationcontent(self):
        self.assertEqual(0, infocontent(1))

    def test_entropy(self):
        x = np.array([0.5, 0.5])
        self.assertEqual(1, entropy(x))

    def test_entropy_whenisnotprobabilityvector_itshouldraiseexception(self):
        with self.assertRaises(Exception):
            x = np.array([0.5, 0.4])
            entropy(x)
        

class TestJointEntropy(unittest.TestCase):

    def setUp(self):
        self.p = np.array([[0.125, 0.0625, 0.03125, 0.03125],
             [0.0625, 0.125, 0.03125, 0.03125],
             [0.0625, 0.0625, 0.0625, 0.0625],
             [0.25,0,0,0]])

    def test_marginalP_0(self):
        npt.assert_array_equal(marginalP(self.p, 0), np.array([0.5  , 0.25 , 0.125, 0.125]))

    def test_marginalY(self):
        npt.assert_array_equal(marginalY(self.p), np.array([0.5  , 0.25 , 0.125, 0.125]))

    def test_marginalX(self):
        npt.assert_array_equal(marginalX(self.p), np.array([0.25  , 0.25 , 0.25, 0.25]))

    def test_marginalP_1(self):
        npt.assert_array_equal(marginalP(self.p, 1), np.array([0.25  , 0.25 , 0.25, 0.25]))


class Test_mutualInformation(unittest.TestCase):
    '''
    Tests:
        mutualinformation([0.2, 0.3; 0.1, 0.4]) and validating that you get the result 0.0349 bits
        mutualinformation([0.5, 0; 0, 0.5]) and validating that you get the result 1 bit
        mutualinformation([0.25, 0.25; 0.25, 0.25]) and validating that you get the result 0 bits
    '''
    def setUp(self):
        self.p1 = np.array([[0.2, 0.3],[0.1, 0.4]])
        self.p2 = np.array([[0.5, 0],[0, 0.5]])
        self.p3 = np.array([[0.25, 0.25],[0.25, 0.25]])

    def test_mutualInformation_1(self):
        self.assertAlmostEqual(mutualInformation(self.p1), 0.0349, 4)  
        self.assertEqual(mutualInformation(self.p2), 1)
        self.assertEqual(mutualInformation(self.p3), 0)



if __name__ == '__main__':
    unittest.main()     