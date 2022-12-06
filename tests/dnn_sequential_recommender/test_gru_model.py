import unittest


class TestGRUModel(unittest.TestCase):
    def test_gru_model(self):
        from aprec.recommenders.sequential.models.gru4rec import GRU4Rec
        from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        USER_ID ='120'

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

if __name__ == "__main__":
    unittest.main()

