import numpy as np
from numpy import random
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import LastItemSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.random_splitter import RandomSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import RecencySequenceSampling, linear_importance
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import exponential_importance
from aprec.evaluation.samplers.random_sampler import RandomTargetItemSampler
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.losses.bpr import BPRLoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from tensorflow.keras.optimizers import Adam

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.0]

def top_recommender():
    return TopRecommender()


def svd_recommender(k):
    return SvdRecommender(k)


def lightfm_recommender(k, loss):
    return LightFMRecommender(k, loss, num_threads=32)


def dnn(model_arch, loss, sequence_splitter, 
                val_sequence_splitter=LastItemSplitter, 
                 target_builder=FullMatrixTargetsBuilder,
                optimizer=Adam(),
                training_time_limit=1200, metric=KerasNDCG(40), 
                max_epochs=10000
                ):
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
                                                          debug=False
                                                          )

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

HISTORY_LEN=200

recommenders = {
    "SASRec-vanilla": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, 
                            dropout_rate=0.2,
                            num_heads=1,
                            num_blocks=2,
                            vanilla=True, 
                            embedding_size=50,
                    ),
            BCELoss(),
            ShiftedSequenceSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN), 
            metric=BCELoss(),
            ),

    "GRU4rec-continuation-bce": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BCELoss(),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-continuation-bce": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BCELoss(),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-continuation-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

        
    "GRU4rec-random-bce": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BCELoss(),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-random-bce": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BCELoss(),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-random-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "GRU4rec-rssExp-bce": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-rssExp-bce": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-rssExp-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "GRU4rec-rssLinear-bce": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-rssLinear-bce": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-rssLinear-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

#bpr
    "GRU4rec-continuation-bpr": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BPRLoss(pred_truncate=4000),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-continuation-bpr": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BPRLoss(pred_truncate=4000),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-continuation-bpr": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BPRLoss(pred_truncate=4000),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

        
    "GRU4rec-random-bpr": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BPRLoss(pred_truncate=4000),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-random-bpr": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BPRLoss(pred_truncate=4000),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-random-bpr": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BPRLoss(pred_truncate=4000),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "GRU4rec-rssExp-bpr": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BPRLoss(pred_truncate=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-rssExp-bpr": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BPRLoss(pred_truncate=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-rssExp-bpr": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BPRLoss(pred_truncate=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "GRU4rec-rssLinear-bpr": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            BPRLoss(pred_truncate=4000),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-rssLinear-bpr": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            BPRLoss(pred_truncate=4000),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-rssLinear-bpr": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BPRLoss(pred_truncate=4000),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

#LambdaRank
    "GRU4rec-continuation-lambdarank": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-continuation-lambdarank": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-continuation-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            LastItemSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

        
    "GRU4rec-random-lambdarank": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-random-lambdarank": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-random-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            RandomSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "GRU4rec-rssExp-lambdarank": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-rssExp-lambdarank": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-rssExp-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "GRU4rec-rssLinear-lambdarank": lambda: dnn(
            GRU4Rec(max_history_len=HISTORY_LEN),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Caser-rssLinear-lambdarank": lambda: dnn(
            Caser(max_history_len=HISTORY_LEN, requires_user_id=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),
    "Sasrec-rssLinear-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=4000),
            lambda: RecencySequenceSampling(0.2, linear_importance()),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(40), 
            ),

    "top": top_recommender, 
    "mf-bpr": lambda: lightfm_recommender(128, 'bpr'),
}


for i in range(1000):
    dropout = random.random() 
    lambdarank_truncate = int(np.random.choice([5000]))
    pct = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) 
    bias = np.random.choice([0.0, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2]) 
    name = f"SASRec-bias:{bias}-dropout:{dropout}-pct:{pct}-truncate:{lambdarank_truncate}"
    func = lambda dropout_rate=dropout,\
         bias=bias, max_pct=pct, truncate=lambdarank_truncate : dnn(
            SASRec(max_history_len=HISTORY_LEN, 
                            dropout_rate=dropout_rate,
                            num_heads=1,
                            num_blocks=2,
                            embedding_size=50,
                    ),
            LambdaGammaRankLoss(pred_truncate_at=truncate),
            lambda: RecencySequenceSampling(max_pct=max_pct, recency_importance=exponential_importance(bias)),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(10),
            )
    recommenders[name] = func


METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = RandomTargetItemSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result
