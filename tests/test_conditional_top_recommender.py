import unittest

from typing import List
from aprec.api.action import Action
from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.recommenders.conditional_top_recommender import ConditionalTopRecommender
from aprec.utils.generator_limit import generator_limit


class TestTopRecommender(unittest.TestCase):
    def test_top_recommender(self):
        recommender = ConditionalTopRecommender(conditional_field='country_id')
        actions: List[Action] = [
            Action(user_id=0, item_id=0, timestamp=0, data={'country_id': 100}),
            Action(user_id=0, item_id=0, timestamp=10, data={'country_id': 100}),
            Action(user_id=0, item_id=1, timestamp=20, data={'country_id': 100}),
        ]
        for action in actions:
            recommender.add_action(action)
        recommender.rebuild_model()
        recommendations = recommender.get_next_items(0, 1)
        self.assertEqual(recommendations, [(0, 2)])

if __name__ == "__main__":
    unittest.main()
