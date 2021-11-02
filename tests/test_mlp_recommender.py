from aprec.recommenders.mlp import GreedyMLP
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens import get_movielens_actions, get_movies_catalog
from aprec.utils.generator_limit import generator_limit
import unittest

USER_ID = '120' 

class TestMLPRecommender(unittest.TestCase):
    def test_mlp_recommender(self):
        mlp_recommender = GreedyMLP(train_epochs=10)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()

