from aprec.recommenders.mlp_historical_embedding import GreedyMLPHistoricalEmbedding
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens import get_movielens_actions
from aprec.utils.generator_limit import generator_limit
from aprec.recommenders.losses.xendcg import XENDCGLoss
import unittest

USER_ID = '120'

class TestMLPRecommenderWithXENDCG(unittest.TestCase):
    def test_mlp_recommender_with_xendcg(self):
        val_users = 10
        batch_size = 10
        mlp_recommender = GreedyMLPHistoricalEmbedding(train_epochs=10, n_val_users=val_users,
                                                       batch_size=batch_size, loss='xendcg')
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        batch_size = mlp_recommender.batch_size
        n_items = mlp_recommender.items.size()
        n_users = mlp_recommender.users.size()
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)



if __name__ == "__main__":
    unittest.main()
