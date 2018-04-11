from aprec.recommenders.constant_recommender import ConstantRecommender
import unittest

class TestConstantRecommender(unittest.TestCase):
    def test_constant_recommender(self):
        constant_recommender = ConstantRecommender(((1, 1),(2, 0.5), (3, 0.4)))
        self.assertEqual(constant_recommender.get_next_items(1, 2), ((1, 1), (2, 0.5)))

if __name__ == "__main__":
    unittest.main()
