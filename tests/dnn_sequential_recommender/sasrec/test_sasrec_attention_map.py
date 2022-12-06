import unittest

class TestSasrecModel(unittest.TestCase):
    def test_sasrec_model(self):
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecModelBuilder
        from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.sequential.targetsplitters.last_item_splitter import SequenceContinuation
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit
        USER_ID = '120'

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        model = SASRecModelBuilder(embedding_size=32)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=3, sequence_splitter=SequenceContinuation)
        recommender.set_val_users(val_users)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        input_seq = recommender.get_model_inputs('120')[0]
        seq, attn = recommender.model.get_seq_embedding(input_seq)
        print(attn[0])
        pass
        

if __name__ == "__main__":
    unittest.main()
    