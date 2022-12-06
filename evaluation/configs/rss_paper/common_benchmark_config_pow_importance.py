from aprec.recommenders.sequential.models.sasrec.sasrec import SASRecModelBuilder
from aprec.recommenders.sequential.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.sequential.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from aprec.recommenders.sequential.targetsplitters.recency_sequence_sampling import pow_importance
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT


from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]
def dnn(model_arch, loss, sequence_splitter, 
                 target_builder=FullMatrixTargetsBuilder,
                training_time_limit=3600,  
                max_epochs=10000):
    from aprec.recommenders.sequential.sequential_recommender import DNNSequentialRecommender

    from tensorflow.keras.optimizers import Adam
    optimizer=Adam(beta_2=0.98)
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          batch_size=1024,
                                                          max_batches_per_epoch=48,
                                                          early_stop_epochs=max_epochs,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          data_generator_processes=4)
HISTORY_LEN=50

recommenders = {
    f"Sasrec-rss-lambdarank-pow:{p}": lambda p=p: dnn(
            SASRecModelBuilder(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, pow_importance(p)),
            target_builder=FullMatrixTargetsBuilder, 
    ) for p in [4, 5, 6]
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

