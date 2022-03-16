from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.first_order_mc import FirstOrderMarkovChainRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
import unittest

USER_ID = '120'

class TestFirstOrderMCRecommender(unittest.TestCase):
    def test_first_order_mc_recommender(self):
        recommender = FilterSeenRecommender(FirstOrderMarkovChainRecommender())
        for action in generator_limit(get_movielens20m_actions(), 100000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()
