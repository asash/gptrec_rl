import unittest
from aprec.datasets.movielens20m import get_movies_catalog
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.default_history_vectorizer import DefaultHistoryVectrizer
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation

from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling, exponential_importance

def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter, 
                target_builder,
                training_time_limit=5,  
                max_epochs=10000, 
                metric = None, 
                pred_history_vectorizer = DefaultHistoryVectrizer()
                ):
    from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender

    from tensorflow.keras.optimizers import Adam
    from aprec.recommenders.metrics.ndcg import KerasNDCG
    if metric is None:
        metric=KerasNDCG(40)
    optimizer=Adam(beta_2=0.98)
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          early_stop_epochs=100,
                                                          batch_size=5,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          pred_history_vectorizer=pred_history_vectorizer,
                                                          debug=True
                                                          )

def sasrec_rss(recency_importance, add_cls=False):
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance), add_cls=add_cls)
        val_splitter = lambda: SequenceContinuation(add_cls=add_cls)
        pred_history_vectorizer = AddMaskHistoryVectorizer() if add_cls else DefaultHistoryVectrizer()
        return dnn(
            SASRec(max_history_len=50, vanilla=False, 
                   num_heads=1, pos_emb_comb='mult', pos_embedding='exp',
                   pos_smoothing=1),
            LambdaGammaRankLoss(pred_truncate_at=1000),
            val_sequence_splitter=val_splitter,
            sequence_splitter=target_splitter,
            target_builder=FullMatrixTargetsBuilder, 
            pred_history_vectorizer=pred_history_vectorizer
            )

class TestSasrecRss(unittest.TestCase):
    def test_sasrec_model_sampled_target_cls(self):
        self.run_model(True)

    def run_model(self, add_cls):
        USER_ID='10'
        from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
        from aprec.datasets.movielens20m import get_movielens20m_actions
        from aprec.utils.generator_limit import generator_limit

        val_users = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        recommender = sasrec_rss(0.8, add_cls) 
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
