from aprec.evaluation.configs.bert4rec_ft.common_benchmark_config import EMBEDDING_SIZE
from aprec.evaluation.metrics.entropy import Entropy
from aprec.evaluation.metrics.highest_score import HighestScore
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.model_confidence import Confidence
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.losses.mean_ypred_loss import MeanPredLoss
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


USERS_FRACTIONS = [1.0]

EXTRA_VAL_METRICS = [HIT(10), NDCG(10), HighestScore(), 
                     Confidence('Softmax'), Confidence('Sigmoid'), 
                     Entropy('Sigmoid', 10),Entropy('Softmax', 10),
                     HIT(100), 
                     MRR(),
                     HIT(1)
                     ]
SEQUENCE_LENGTH=50

def sasrec_rss_lambdarank(recency_importance): 
        from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
        from aprec.recommenders.sequential.target_builders.positives_only_targets_builder import PositvesOnlyTargetBuilder
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance))
        loss='lambdarank'
        loss_parameters=dict(pred_truncate_at=4000)
        model_config = SASRecConfig(loss=loss, loss_params=loss_parameters)
        return model(
            model_config,
            sequence_splitter=target_splitter,
            target_builder=PositvesOnlyTargetBuilder)

def caser_rss_bce(recency_importance): 
        from aprec.recommenders.sequential.models.caser import CaserConfig
        from aprec.recommenders.sequential.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance
        from aprec.losses.bce import BCELoss
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance))
        model_config = CaserConfig()
        return model(
            model_config,
            loss=BCELoss(),
            sequence_splitter=target_splitter,
            target_builder=FullMatrixTargetsBuilder)

def gru4rec_lambdarank(recency_importance): 
        from aprec.recommenders.sequential.models.gru4rec import GRU4RecConfig
        from aprec.recommenders.sequential.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
        from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import exponential_importance
        from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
        target_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance))
        model_config = GRU4RecConfig()
        return model(
            model_config,
            loss=LambdaGammaRankLoss(pred_truncate_at=4000),
            sequence_splitter=target_splitter,
            target_builder=FullMatrixTargetsBuilder)


def model(model_config, sequence_splitter, 
                target_builder,
                loss=MeanPredLoss()
                ):
    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig

    config = SequentialRecommenderConfig(model_config,                       
                                batch_size=128,
                                loss=loss,
                                early_stop_epochs=100000,
                                training_time_limit=3600,
                                max_batches_per_epoch=256,
                                sequence_splitter=sequence_splitter, 
                                targets_builder=target_builder, 
                                use_keras_training=True,
                                extra_val_metrics=EXTRA_VAL_METRICS, 
                                sequence_length=SEQUENCE_LENGTH
                                )
    return SequentialRecommender(config)

recommenders = {
}
for rss in [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    recommenders[f'sasrec_lambdarank_rss:{rss}'] = lambda recency_importance=rss: sasrec_rss_lambdarank(recency_importance)
    recommenders[f'caser_bce_rss:{rss}'] = lambda recency_importance=rss: caser_rss_bce(recency_importance)
    recommenders[f'gru4rec_lambdarank_rss:{rss}'] = lambda recency_importance=rss: gru4rec_lambdarank(recency_importance)

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]

def get_recommenders(filter_seen: bool, filter_recommenders = set()):
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
                continue
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

