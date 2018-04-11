from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.recommenders.top_recommender import TopRecommender
from aprec.utils.generator_limit import generator_limit

import unittest
class TestTopRecommender(unittest.TestCase):
    def test_top_recommender(self):
        recommender = TopRecommender()
        catalog = get_movies_catalog()
        for action in generator_limit(get_movielens_actions(), 1000):
            recommender.add_action(action)
        recommendations = recommender.get_next_items(1, 5) 
        self.assertEqual(recommendations, [('260', 10), ('589', 9), ('1196', 8), ('480', 8), ('1', 8)])

if __name__ == "__main__":
    unittest.main()
