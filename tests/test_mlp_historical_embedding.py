from aprec.recommenders.mlp_historical_embedding import GreedyMLPHistoricalEmbedding
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
import unittest

USER_ID = '120'

class TestMLPHistoricalEmbeddingRecommender(unittest.TestCase):
    def test_mlp_recommender(self):

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        mlp_recommender = GreedyMLPHistoricalEmbedding(train_epochs=10, early_stop_epochs=5,
                                                       batch_size=5)
        mlp_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        print(recs)


    def test_gru_model(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        recommender = GreedyMLPHistoricalEmbedding(train_epochs=10, early_stop_epochs=5,
                                                       batch_size=5, model_arch='gru', training_time_limit=10)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        print(recs)



if __name__ == "__main__":
    unittest.main()
