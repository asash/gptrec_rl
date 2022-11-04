from aprec.evaluation.split_actions import LeaveOneOut

from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT


from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def top_recommender():
     from aprec.recommenders.top_recommender import TopRecommender
     return TopRecommender()


def svd_recommender(k):
    from aprec.recommenders.svd import SvdRecommender
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    from aprec.recommenders.lightfm import LightFMRecommender
    return LightFMRecommender(k, loss, num_threads=32)


def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter=SequenceContinuation, 
                 target_builder=FullMatrixTargetsBuilder,
                training_time_limit=3600,  
                max_epochs=10000, 
                metric = None):
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
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          debug=True
                                                          )

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

HISTORY_LEN=50

def vanilla_sasrec():
    model_arch = SASRec(max_history_len=HISTORY_LEN, vanilla=True, num_heads=1)

    return dnn(model_arch,  BCELoss(),
            ShiftedSequenceSplitter,
            target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN), 
            metric=BCELoss())

def sasrec_rss(recency_importance):
        return dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False, num_heads=1),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(recency_importance)),
            target_builder=FullMatrixTargetsBuilder)

    

recommenders = {
    "SASRec-vanilla": vanilla_sasrec,
    "Sasrec-rss-lambdarank-0.01": lambda: sasrec_rss(0.01),
    "Sasrec-rss-lambdarank-0.1": lambda: sasrec_rss(0.1),
    "Sasrec-rss-lambdarank-0.2": lambda: sasrec_rss(0.2),
    "Sasrec-rss-lambdarank-0.3": lambda: sasrec_rss(0.3),
    "Sasrec-rss-lambdarank-0.4": lambda: sasrec_rss(0.4),
    "Sasrec-rss-lambdarank-0.5": lambda: sasrec_rss(0.5),
    "Sasrec-rss-lambdarank-0.6": lambda: sasrec_rss(0.6),
    "Sasrec-rss-lambdarank-0.7": lambda: sasrec_rss(0.7),
    "Sasrec-rss-lambdarank-0.9": lambda: sasrec_rss(0.9),
    "Sasrec-rss-lambdarank-0.99": lambda: sasrec_rss(0.99),
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


DATASET = "ml-20m_warm5"
N_VAL_USERS=1024
#MAX_TEST_USERS=138493
MAX_TEST_USERS=4096
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)