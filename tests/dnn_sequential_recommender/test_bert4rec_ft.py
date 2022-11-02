import unittest

from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.recommenders.metrics.ndcg import KerasNDCG
class TestOwnBERT4rec(unittest.TestCase):
    def test_bert4rec_ft_recommender(self):
        from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_with_negatives import ItemsMaskingWithNegativesTargetsBuilder
        from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_samplers import RandomNegativesWithCosSimValues
        from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.tests.dnn_sequential_recommender.test_dnn_sequential import USER_ID
        from aprec.utils.generator_limit import generator_limit
        from aprec.datasets.movielens20m import get_movielens20m_actions, get_movies_catalog
        from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
        from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
        from aprec.recommenders.dnn_sequential_recommender.models.bert4recft.bert4recft import BERT4RecFT
        from aprec.losses.items_masking_loss_proxy import ItemsMaksingLossProxy


        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        sequence_len = 100
        add_positive = True
        negatives_per_positive = 5
        model = BERT4RecFT(embedding_size=32, max_history_len=sequence_len, output_layer_activation='softmax')
        batch_size = 5
        n_targets = negatives_per_positive
        if add_positive:
            n_targets += 1
        metric = ItemsMaksingLossProxy(KerasNDCG(n_targets), negatives_per_positive, sequence_len, add_positive=add_positive)
        metric.set_batch_size(batch_size)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=50000,
                                               batch_size=batch_size,
                                               training_time_limit=20, 
                                               loss = ItemsMaksingLossProxy(LambdaGammaRankLoss(), negatives_per_positive, sequence_len, add_positive=add_positive),
                                               debug=True, sequence_splitter=lambda: ItemsMasking(), 
                                               targets_builder= lambda: ItemsMaskingWithNegativesTargetsBuilder(relative_positions_encoding=True, 
                                                                                 add_positive=add_positive,                               
                                                                                 negatives_sampler=RandomNegativesWithCosSimValues(negatives_per_positive)),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=metric,
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               max_batches_per_epoch=5, 
                                               )
        recommender.set_val_users(val_users)
        recommender = FilterSeenRecommender(recommender)
        for action in generator_limit(get_movielens20m_actions(), 10000):
            recommender.add_action(action)
        recommender.rebuild_model()
        recs = recommender.recommend(USER_ID, 10)
        catalog = get_movies_catalog()
        for rec in recs:
            print(catalog.get_item(rec[0]), "\t", rec[1])


if __name__ == "__main__":
    unittest.main()