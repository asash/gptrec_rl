from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.datasets.movielens import get_movielens_actions
from aprec.utils.generator_limit import generator_limit
import unittest

USER_ID = '120'

class TestLightFMRecommender(unittest.TestCase):
    def test_mlp_recommender(self):
        lightfm_recommender = LightFMRecommender(30, 'bpr')
        recommender = FilterSeenRecommender(lightfm_recommender)
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()