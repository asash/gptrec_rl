import unittest

class TestVit4rec(unittest.TestCase):
    def test_vit4rec(self):

        from aprec.recommenders.dnn_sequential_recommender.models.vit4rec import Vit4Rec
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = Vit4Rec()
        recommender = DNNSequentialRecommender(model, train_epochs=10, early_stop_epochs=5,
                                               batch_size=5, training_time_limit=100)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend('120', 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()