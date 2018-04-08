from aprec.evaluation.metrics.precision import precision
import unittest

class TestPrecision(unittest.TestCase):
    def test_precsion(self):
        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [1, 3]
        self.assertEqual(precision(recommended, actual, 1), 1)
        self.assertEqual(precision(recommended, actual, 2), 0.5)
        self.assertEqual(precision(recommended, actual, 3), 2/3)

if __name__ == "__main__":
    unittest.main()
