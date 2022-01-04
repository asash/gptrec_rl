import numpy as np
from aprec.evaluation.samplers.random_sampler import RandomTargetItemSampler
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.losses.bpr import BPRLoss
from aprec.losses.top1 import TOP1Loss
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.biased_percentage_splitter import BiasedPercentageSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import LastItemSplitter
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import ShiftedSequenceSplitter
from aprec.recommenders.metrics.ndcg import KerasNDCG
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender

from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser

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


def dnn(model_arch, loss, sequence_splitter, target_builder=FullMatrixTargetsBuilder(),
         learning_rate=0.001, training_time_limit=3600, metric=KerasNDCG(40)):
    return DNNSequentialRecommender(train_epochs=10000, loss=loss,
                                                          model_arch=model_arch,
                                                          optimizer=Adam(learning_rate),
                                                          early_stop_epochs=100,
                                                          batch_size=128,
                                                          training_time_limit=training_time_limit,
                                                          sequence_splitter=sequence_splitter, 
                                                          targets_builder=target_builder, 
                                                          metric=metric,
                                                          )

def salrec(loss, training_time_limit=3600, target_decay=1.0, seq_len = 100):
    return SalrecRecommender(loss=loss, target_decay=target_decay, max_history_len=seq_len, training_time_limit=training_time_limit,
                             train_epochs=10000, early_stop_epochs=1000, num_blocks=3, num_bottlenecks=1)


def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender


recommenders = {
    "SASRec": lambda: dnn(
            SASRec(max_history_len=200, dropout_rate=0.2, num_heads=2, vanilla=True),
            BCELoss(),
            ShiftedSequenceSplitter(),
            target_builder=NegativePerPositiveTargetBuilder(200), 
            metric=BCELoss()
            ),
    "top": top_recommender, 
    "mf-bpr": lightfm_recommender(128, 'bpr')
}
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
