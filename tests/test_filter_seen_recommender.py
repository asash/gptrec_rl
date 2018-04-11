from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.api.action import Action

import unittest

class TestConstantRecommender(unittest.TestCase):
    def test_constant_recommender(self):
        constant_recommender = ConstantRecommender(((1, 1),(2, 0.5), (3, 0.4)))
        recommender = FilterSeenRecommender(constant_recommender)
        recommender.add_action(Action(user_id=1, item_id=2, timestamp=1))
        self.assertEqual(recommender.get_next_items(1, 2), [(1, 1), (3, 0.4)])

if __name__ == "__main__":
    unittest.main()
