import os
import unittest

from aprec.recommenders.sequential.targetsplitters.fair_item_masking import FairItemMasking
from aprec.api.items_ranking_request import ItemsRankingRequest

class TestBERT4RecFairness(unittest.TestCase):
    def setUp(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    def test_bert4rec_fairness(self):
        from aprec.recommenders.sequential.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance
        from aprec.recommenders.sequential.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.sequential.models.bert4rec.bert4rec import BERT4Rec
        from aprec.losses.mean_ypred_ploss import MeanPredLoss

        USER_ID = '120'

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        embedding_size=32
        model = BERT4Rec(embedding_size=embedding_size)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=5,
                                               training_time_limit=10, 
                                               loss = MeanPredLoss(),
                                               sequence_splitter=lambda: FairItemMasking(), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=True),
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               eval_batch_size=8
                                               )
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        ranking_request = ItemsRankingRequest('120', ['608', '294', '648'])
        recommender.add_test_items_ranking_request(ranking_request)
        batch1 = [('120', None), ('10', None)]
        recs1 = recommender.recommender.recommend_multiple(batch1, 10)        

        batch2 = [(str(i), None) for i in range(1, 25)]
        batch_result = recommender.recommend_batch(batch2, 10)
        one_by_one_result = []
        for user_id, features in batch2:
            one_by_one_result.append(recommender.recommend(user_id, 10))
            
        for i in range(len(batch2)):
            for j in range(len(batch_result[i])):
                batch_item, batch_score = batch_result[i][j]
                one_by_one_item, one_by_one_score = one_by_one_result[i][j]
                self.assertEquals(batch_item, one_by_one_item)
                self.assertAlmostEquals(batch_score, one_by_one_score, places=4)

        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])

if __name__ == "__main__":
    unittest.main()

