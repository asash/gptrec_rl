from aprec.recommenders.dnn_recommender import DNNRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
from aprec.losses.lambdarank import LambdaRankLoss
import unittest

USER_ID = '120'

class TestMLPRecommenderWithLambdarank(unittest.TestCase):
    def test_mlp_recommender_with_lambdarank(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        batch_size = 10
        mlp_recommender = DNNRecommender(train_epochs=10, batch_size=batch_size,
                                         output_layer_activation='linear')
        mlp_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        batch_size = mlp_recommender.batch_size
        n_items = mlp_recommender.items.size()
        n_users = mlp_recommender.users.size()
        loss = LambdaRankLoss(n_items, min(batch_size, n_users - len(val_users)), ndcg_at=10)
        mlp_recommender.set_loss(loss)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        metadata = recommender.get_metadata()
        print(metadata)



if __name__ == "__main__":
    unittest.main()
