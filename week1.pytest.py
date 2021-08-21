import pytest
import numpy as np
from week1 import infocontent, entropy1, marginalP, marginalX, marginalY

class TestEntropy(unittest.TestCase):

    def test_informationcontent(self):
        self.assertEqual(0, infocontent(1))

    def test_entropy(self):
        x = np.array([0.5, 0.5])
        self.assertEqual(1, entropy1(x))

    def test_entropy_whenisnotprobabilityvector_itshouldraiseexception(self):
        with self.assertRaises(Exception):
            x = np.array([0.5, 0.4])
            entropy1(x)
        

class TestJointEntropy(unittest.TestCase):

    def test_jointEntropy(self):
        p = np.array([[0.125, 0.0625, 0.03125, 0.03125],
             [0.0625, 0.125, 0.03125, 0.03125],
             [0.0625, 0.0625, 0.0625, 0.0625],
             [0.25,0,0,0]])
        self.assertSequenceEqual(marginalP(p, 0), [0.5  , 0.25 , 0.125, 0.125], np.array)


if __name__ == '__main__':
    unittest.main()     