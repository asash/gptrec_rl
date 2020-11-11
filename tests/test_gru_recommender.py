import unittest

from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.gru_recommender import GRURecommender
from aprec.utils.generator_limit import generator_limit

USER_ID = '120'

class TestMLPRecommender(unittest.TestCase):
    def test_mlp_recommender(self):
        mlp_recommender = GRURecommender(train_epochs=1, n_val_users=10, batch_size=10)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()
