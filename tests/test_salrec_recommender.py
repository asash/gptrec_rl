from aprec.api.user import User
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets import mts_kion
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
import tensorflow as tf
import unittest

USER_ID = '120'


class TestSalrecRecommender(unittest.TestCase):
    def test_salrec_recommender(self):
        batch_size = 1
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        salrec_recommender = SalrecRecommender(train_epochs=10, batch_size=batch_size,
                                               output_layer_activation='linear',
                                               max_history_len=50,
                                               loss='lambdarank',
                                               loss_internal_dtype=tf.float16,
                                               loss_lambda_normalization=False)
        salrec_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(salrec_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)

        recs = recommender.recommend('aadfadsf', 10)
        print(recs)

    def test_salrec_recommender_unordered(self):
        batch_size = 10
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        salrec_recommender = SalrecRecommender(train_epochs=10, batch_size=batch_size,
                                               output_layer_activation='linear', max_history_len=50, positional=False)
        salrec_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(salrec_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        batch_size = salrec_recommender.batch_size
        n_items = salrec_recommender.items.size()
        n_users = salrec_recommender.users.size()
        loss = LambdaGammaRankLoss(n_items, min(batch_size, n_users - len(val_users)), ndcg_at=10)
        salrec_recommender.set_loss(loss)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)

    def test_salrec_time_limit(self):
        batch_size = 10
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        salrec_recommender = SalrecRecommender(train_epochs=100000, batch_size=batch_size,
                                               output_layer_activation='linear',
                                               max_history_len=50, training_time_limit=5)
        salrec_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(salrec_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        batch_size = salrec_recommender.batch_size
        n_items = salrec_recommender.items.size()
        n_users = salrec_recommender.users.size()
        loss = LambdaGammaRankLoss(n_items, min(batch_size, n_users - len(val_users)), ndcg_at=10)
        salrec_recommender.set_loss(loss)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)

    def test_user_feature_hashes(self):
        user = User(user_id='1', cat_features={'age': 10, 'sex': 'F'})
        recommender = SalrecRecommender()
        recommender.add_user(user)
        print(recommender.user_feature_hashes[0])


    def test_with_user_hashes(self):
        all_users = mts_kion.get_users()
        recommender = SalrecRecommender(train_epochs=5, batch_size=10, output_layer_activation='linear',
                                        max_history_len=10, training_time_limit=5)
        for user in all_users:
            recommender.add_user(user)
        actions = mts_kion.get_mts_kion_dataset(max_actions=10000)

        actioned_users = set()
        for action in actions:
            recommender.add_action(action)
            actioned_users.add(action.user_id)

        val_users = sorted(list(actioned_users))[:10]
        recommender.set_val_users(val_users)
        recommender.rebuild_model()


if __name__ == "__main__":
    unittest.main()
