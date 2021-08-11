import unittest
import numpy as np
from week1 import infocontent, entropy

class TestEntropy(unittest.TestCase):

    def test_informationcontent(self):
        self.assertEqual(0, infocontent(1))

    def test_entropy(self):
        x = np.array([0.5, 0.5])
        self.assertEqual(1, entropy(x))

    def test_entropy_whenisnotprobabilityvector_itshouldraiseexception(self):
        x = np.array([0.5, 0.4])
        entropy(x)
        self.assertRaises(Exception)

if __name__ == '__main__':
    unittest.main()                                  