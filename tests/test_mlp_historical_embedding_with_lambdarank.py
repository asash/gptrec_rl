from aprec.recommenders.mlp_historical_embedding import GreedyMLPHistoricalEmbedding
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens import get_movielens_actions
from aprec.utils.generator_limit import generator_limit
from aprec.recommenders.losses.lambdarank import pairwise_loss
import unittest

USER_ID = '120'

class TestMLPRecommenderWithLambdarank(unittest.TestCase):
    def test_mlp_recommender_with_lambdarank(self):
        loss = pairwise_loss
        mlp_recommender = GreedyMLPHistoricalEmbedding(train_epochs=10, n_val_users=10, loss=loss)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        metadata = recommender.get_metadata()
        print(metadata)



if __name__ == "__main__":
    unittest.main()
