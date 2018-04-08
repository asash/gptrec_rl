from aprec.evaluation.metrics.precision import precision
from aprec.api.action import Action
import unittest

class TestPrecision(unittest.TestCase):
    def test_precsion(self):
        recommended = [(1, 2), (2, 1), (3, 0.5)]
        actual = [Action(user_id = 1, item_id = 1, timestamp=1), 
                  Action(user_id = 1, item_id = 3, timestamp=2)]
        self.assertEqual(precision(recommended, actual, 1), 1)
        self.assertEqual(precision(recommended, actual, 2), 0.5)
        self.assertEqual(precision(recommended, actual, 3), 2/3)

if __name__ == "__main__":
    unittest.main()
