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
                                                          early_stop_epochs=None,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          val_sequence_splitter = val_sequence_splitter,
                                                          metric=metric,
                                                          debug=False
                                                          )

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

HISTORY_LEN=50

def vanilla_sasrec():
    model_arch = SASRec(max_history_len=HISTORY_LEN, 
                            dropout_rate=0.2,
                            num_heads=1,
                            num_blocks=2,
                            vanilla=True, 
                            embedding_size=50,
                    )

    return dnn(model_arch,  BCELoss(),
            ShiftedSequenceSplitter,
            target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN), 
            metric=BCELoss(),
            )

def sasrec_lambdarank_time(time):
    return lambda time=time: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
                    LambdaGammaRankLoss(pred_truncate_at=4000),
                    lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
                    target_builder=FullMatrixTargetsBuilder,
                    training_time_limit=time 
            )

recommenders = {
            "Sasrec-rss-lambdarank-1m": sasrec_lambdarank_time(60),
            "Sasrec-rss-lambdarank-2m": sasrec_lambdarank_time(60*2),
            "Sasrec-rss-lambdarank-4m": sasrec_lambdarank_time(60*4),
            "Sasrec-rss-lambdarank-8m": sasrec_lambdarank_time(60*8),
            "Sasrec-rss-lambdarank-16m": sasrec_lambdarank_time(60*16),
            "Sasrec-rss-lambdarank-30m": sasrec_lambdarank_time(60*30),
            "Sasrec-rss-lambdarank-1h": sasrec_lambdarank_time(3600),
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
