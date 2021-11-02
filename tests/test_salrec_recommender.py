from aprec.recommenders.salrec_recommender import SalrecRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens import get_movielens_actions
from aprec.utils.generator_limit import generator_limit
from aprec.recommenders.losses.lambdarank import LambdaRankLoss
import tensorflow as tf
import unittest

USER_ID = '120'

class TestSalrecRecommender(unittest.TestCase):
    def test_salrec_recommender(self):
        batch_size = 10
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        salrec_recommender = SalrecRecommender(train_epochs=10, batch_size=batch_size,
                                                       output_layer_activation='linear')
        salrec_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(salrec_recommender)
        for action in generator_limit(get_movielens_actions(), 10000):
            recommender.add_action(action)
        batch_size = salrec_recommender.batch_size
        n_items = salrec_recommender.items.size()
        n_users = salrec_recommender.users.size()
        loss = LambdaRankLoss(n_items, min(batch_size, n_users - len(val_users)), 10)
        salrec_recommender.set_loss(loss)
        recommender.rebuild_model()
        recs = recommender.get_next_items(USER_ID, 10)
        metadata = recommender.get_metadata()
        print(metadata)



if __name__ == "__main__":
    unittest.main()
