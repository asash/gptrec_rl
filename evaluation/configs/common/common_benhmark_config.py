import numpy as np
from aprec.evaluation.samplers.random_sampler import RandomTargetItemSampler
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import LastItemSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.biased_percentage_splitter import BiasedPercentageSplitter
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.losses.lambda_gamma_rank import LambdaGammaRankLoss



from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.hit import HIT

from tensorflow.keras.optimizers import Adam

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

USERS_FRACTIONS = [1.]

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
                training_time_limit=3600, metric=KerasNDCG(40), 
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


recommenders = {
    "SASRec-bias:0.8-dropout:0.2-pct:0.2":  lambda dropout_rate=0.2,\
         bias=0.8, max_pct=0.2 : dnn(
            SASRec(max_history_len=200, 
                            dropout_rate=dropout_rate,
                            num_heads=1,
                            num_blocks=2,
                            embedding_size=50,
                    ),
            LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975),
            lambda: BiasedPercentageSplitter(max_pct=max_pct, bias=bias),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(10),
            ),

    "SASRec": lambda: dnn(
            SASRec(max_history_len=200, 
                            dropout_rate=0.2,
                            num_heads=1,
                            num_blocks=2,
                            vanilla=True, 
                            embedding_size=50,
                    ),
            BCELoss(),
            ShiftedSequenceSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=lambda: NegativePerPositiveTargetBuilder(200), 
            metric=BCELoss(),
            ),
    "top": top_recommender, 
    "mf-bpr": lambda: lightfm_recommender(128, 'bpr'),


}

for i in range(1000):
    dropout_rate = float(np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    max_pct = float(np.random.choice([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]))
    bias  = float(np.random.choice([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]))
    recommenders[f"SASRec-bias:{bias}-dropout:{dropout_rate}-pct:{max_pct}"]= lambda dropout_rate=dropout_rate,\
         bias=bias, max_pct=max_pct : dnn(
            SASRec(max_history_len=200, 
                            dropout_rate=dropout_rate,
                            num_heads=1,
                            num_blocks=2,
                            embedding_size=50,
                    ),
            LambdaGammaRankLoss(pred_truncate_at=2500, bce_grad_weight=0.975),
            lambda: BiasedPercentageSplitter(max_pct=max_pct, bias=bias),
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(10),
            )


for i in range(0):
    loss_type = np.random.choice(["top1max", 'bce', 'lambdarank'])

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40)]
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
