import os
import unittest

from aprec.api.items_ranking_request import ItemsRankingRequest
class TestTwoBerts(unittest.TestCase):
    def setUp(self) -> None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        return super().setUp()
    
    def test_two_berts(self):
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.models.bert4recft.two_berts import TwoBERTS
        from aprec.losses.mean_ypred_ploss import MeanPredLoss

        USER_ID = '120'

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        embedding_size=32
        model = TwoBERTS(embedding_size=embedding_size, num_samples=400)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, 
                                               loss = MeanPredLoss(),
                                               sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=True),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               eval_batch_size=8)
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        ranking_request = ItemsRankingRequest('120', ['608', '294', '648'])
        recommender.add_test_items_ranking_request(ranking_request)
        batch1 = [('120', None), ('10', None)]
        recs = recommender.recommender.recommend_multiple(batch1, 10)        
        catalog = get_movies_catalog()
        for rec in recs[0]:
            print(catalog.get_item(rec[0]), "\t", rec[1])


if __name__ == "__main__":
    unittest.main()