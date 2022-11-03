import unittest

class TestSasrecModel(unittest.TestCase):
    def test_sasrec_model(self):
        from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        USER_ID = '120'

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = SASRec(embedding_size=32)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=3, debug=True, sequence_splitter=SequenceContinuation)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()
    
