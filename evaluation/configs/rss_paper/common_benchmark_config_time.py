from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


USERS_FRACTIONS = [1.0]


def dnn(model_config, loss, sequence_splitter, 
                target_builder,
                training_time_limit=3600,  
                max_epochs=10000,
                sequence_len=50
                ):

    from aprec.recommenders.sequential.sequential_recommender import SequentialRecommender
    from aprec.recommenders.sequential.sequential_recommender_config import SequentialRecommenderConfig
    recommender_config = SequentialRecommenderConfig(model_config=model_config,
                                                         train_epochs=max_epochs, loss=loss,
                                                          early_stop_epochs=100000000,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder,
                                                          sequence_length=sequence_len
                                                          )
    return SequentialRecommender(recommender_config)


def sasrec_lambdarank_time(time):
    from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
    from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecConfig
    from aprec.recommenders.sequential.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
    from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling, exponential_importance
    from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
    return lambda time=time: dnn(
                    SASRecConfig(vanilla=False),
                    LambdaGammaRankLoss(pred_truncate_at=4000),
                    lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
                    target_builder=FullMatrixTargetsBuilder,
                    training_time_limit=time 
            )

recommenders = {
#            "Sasrec-rss-lambdarank-1m": sasrec_lambdarank_time(60),
#            "Sasrec-rss-lambdarank-2m": sasrec_lambdarank_time(60*2),
#            "Sasrec-rss-lambdarank-4m": sasrec_lambdarank_time(60*4),
#            "Sasrec-rss-lambdarank-8m": sasrec_lambdarank_time(60*8),
#            "Sasrec-rss-lambdarank-16m": sasrec_lambdarank_time(60*16),
#            "Sasrec-rss-lambdarank-30m": sasrec_lambdarank_time(60*30),
#            "Sasrec-rss-lambdarank-1h": sasrec_lambdarank_time(3600),
            "Sasrec-rss-lambdarank-2h": sasrec_lambdarank_time(3600*2),
            "Sasrec-rss-lambdarank-4h": sasrec_lambdarank_time(3600*4),
            "Sasrec-rss-lambdarank-8h": sasrec_lambdarank_time(3600*8),
            "Sasrec-rss-lambdarank-16h": sasrec_lambdarank_time(3600*16),
     }

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
