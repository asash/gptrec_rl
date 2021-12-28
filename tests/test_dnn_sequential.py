from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.utils.generator_limit import generator_limit
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.losses.xendcg import XENDCGLoss
import aprec.datasets.mts_kion as kion
import unittest

from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.models.mlp_sequential import SequentialMLPModel
from aprec.recommenders.dnn_sequential_recommender.featurizers.hashing_featurizer import HashingFeaturizer
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec_kion import KionChallengeSASRec

USER_ID = '120'

class TestDnnSequentialRecommender(unittest.TestCase):
    def test_mlp_recommender(self):
        model = SequentialMLPModel()
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        mlp_recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                                   batch_size=5)
        mlp_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)


    def test_gru_model(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = GRU4Rec()
        recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=10)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)


    def test_sasrec_model(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = SASRec(embedding_size=32)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, debug=True, train_on_last_item_only=True)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

    def test_sasrec_model_no_reuse(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = SASRec(embedding_size=32, reuse_item_embeddings=False, encode_output_embeddings=True)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, debug=True, train_on_last_item_only=True)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)



    def test_sasrec_kion(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        user_featurizer = HashingFeaturizer()
        model = KionChallengeSASRec()
        recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=10,
                                               users_featurizer=user_featurizer,
                                               debug=False
                                               )
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        users = kion.get_users()
        for action in kion.get_mts_kion_dataset(1000):
            recommender.add_action(action)
        for user in users:
            recommender.add_user(user)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)


    def test_caser_model(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = Caser()
        recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=10)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

    def test_caser_extra_features(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        user_featurizer = HashingFeaturizer()
        item_featurizer = HashingFeaturizer()
        model = Caser(user_extra_features=True, requires_user_id=False)
        recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=10,
                                               users_featurizer=user_featurizer,
                                               items_featurizer=item_featurizer
                                               )
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        users = kion.get_users()
        items = kion.get_items()
        for action in kion.get_mts_kion_dataset(1000):
            recommender.add_action(action)
        for user in users:
            recommender.add_user(user)

        for item in items:
            recommender.add_item(item)

        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)


    def test_caser_model_no_uid(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = Caser(requires_user_id=False)
        recommender = DNNSequentialRecommender(model, train_epochs=10,
                                               early_stop_epochs=5, batch_size=5, training_time_limit=10)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

    def test_mlp_recommender_with_lambdarank(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        batch_size = 10
        model = SequentialMLPModel(output_layer_activation='linear')
        loss = LambdaGammaRankLoss(ndcg_at=10)
        mlp_recommender = DNNSequentialRecommender(model, loss, train_epochs=10, batch_size=batch_size)

        mlp_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        mlp_recommender.set_loss(loss)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)


    def test_mlp_recommender_with_xendcg(self):
        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        batch_size = 10
        model = SequentialMLPModel()
        loss = XENDCGLoss()
        mlp_recommender = DNNSequentialRecommender(model, loss, train_epochs=10,
                                                   batch_size=batch_size, debug=True)
        mlp_recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(mlp_recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)
        metadata = recommender.get_metadata()
        print(metadata)



if __name__ == "__main__":
    unittest.main()
