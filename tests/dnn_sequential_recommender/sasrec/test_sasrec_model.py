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
                                               training_time_limit=3,  sequence_splitter=SequenceContinuation)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)

        recommender.rebuild_model()

        batch2 = [(str(i), None) for i in range(1, 25)]
        batch_result = recommender.recommend_batch(batch2, 10)
        one_by_one_result = []
        for user_id, features in batch2:
            one_by_one_result.append(recommender.recommend(user_id, 10))
        for i in range(len(batch2)):
            for j in range(len(batch_result[i])):
                _, batch_score = batch_result[i][j]
                _, one_by_one_score = one_by_one_result[i][j]
                self.assertAlmostEquals(batch_score, one_by_one_score, places=3)


        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()
    
