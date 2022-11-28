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
                 target_builder=FullMatrixTargetsBuilder,
                training_time_limit=3600,  
                max_epochs=10000):
    from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender

    from tensorflow.keras.optimizers import Adam
    optimizer=Adam(beta_2=0.98)
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=optimizer,
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
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
            )

recommenders = {
    "bert4rec-1h": lambda: vanilla_bert4rec(3600), 
    "top": top_recommender, 
    "mf-bpr": lambda: lightfm_recommender(128, 'bpr'),

    "SASRec-vanilla": vanilla_sasrec,
    

    "GRU4rec-continuation-bce": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BCELoss(),
            SequenceContinuation,
            target_builder=FullMatrixTargetsBuilder, 
            ),

    "Caser-continuation-bce": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN),
            BCELoss(),
            SequenceContinuation,
            target_builder=FullMatrixTargetsBuilder, 
            ),

    "Sasrec-continuation-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            SequenceContinuation,
            target_builder=FullMatrixTargetsBuilder, 
            ),

    "GRU4rec-rss-bce": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            target_builder=FullMatrixTargetsBuilder, 
            ),
    "Caser-rss-bce": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            target_builder=FullMatrixTargetsBuilder, 
            ),
    "Sasrec-rss-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            target_builder=FullMatrixTargetsBuilder, 
            ),


#LambdaRank
    "GRU4rec-continuation-lambdarank": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            SequenceContinuation,
            target_builder=FullMatrixTargetsBuilder, 
            ),
    "Caser-continuation-lambdarank": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            SequenceContinuation,
            target_builder=FullMatrixTargetsBuilder, 
            ),
    "Sasrec-continuation-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            SequenceContinuation,
            target_builder=FullMatrixTargetsBuilder, 
            ),

        
    "GRU4rec-rss-lambdarank": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            target_builder=FullMatrixTargetsBuilder, 
            ),

    "Caser-rss-lambdarank": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            target_builder=FullMatrixTargetsBuilder, 
            ),

    "Sasrec-rss-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            target_builder=FullMatrixTargetsBuilder, 
            )
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
