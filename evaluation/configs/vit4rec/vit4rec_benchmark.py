
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.losses.bce import BCELoss
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


USERS_FRACTIONS = [1.0]

def vit4rec():
    from tensorflow.keras.optimizers import Adam
    from aprec.recommenders.dnn_sequential_recommender.models.vit4rec import Vit4Rec
    from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
    from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
    from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
    from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
    from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
    from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss
    from aprec.recommenders.metrics.ndcg import KerasNDCG
    model_arch= Vit4Rec()
    val_sequence_splitter=SequenceContinuation 
    loss=BCELoss()
    sequence_splitter = lambda: RecencySequenceSampling(0.2, exponential_importance(0.8))
    target_builder=FullMatrixTargetsBuilder
    optimizer=Adam(beta_2=0.98)
    training_time_limit=360000 
    metric=KerasNDCG(40) 
    max_epochs=10000

    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          early_stop_epochs=200,
                                                          batch_size=16,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          debug=False, 
                                                          max_batches_per_epoch=128,
                                                          )
HISTORY_LEN=50

recommenders = {
    "VIT4Rec-RSS-Lambdarank": vit4rec
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

DATASET = "BERT4rec.ml-1m"
N_VAL_USERS=2048
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)
