import unittest


class TestSasrecSampledTarget(unittest.TestCase):
    def test_sasrec_model_sampled_target(self):
        USER_ID='10'
        from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
        from aprec.recommenders.dnn_sequential_recommender.target_builders.sampled_matrix_target_builder import SampledMatrixBuilder
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = SASRec(embedding_size=32, sampled_targets=101)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=7, 
                                               sequence_splitter=SequenceContinuation, 
                                               targets_builder=SampledMatrixBuilder)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        print(recs)

if __name__ == "__main__":
    unittest.main()

