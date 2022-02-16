import numpy as np
from numpy import random
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.losses.mean_ypred_ploss import MeanPredLoss
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
from aprec.recommenders.dnn_sequential_recommender.models.deberta4rec.deberta4rec import Deberta4Rec
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.dnn_sequential_recommender.models.gru4rec import GRU4Rec
from aprec.recommenders.dnn_sequential_recommender.models.caser import Caser
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
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



def deberta4rec(relative_position_encoding, sequence_len):
        model = Deberta4Rec(embedding_size=64, intermediate_size=128, num_hidden_layers=2, max_history_len=sequence_len)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=256,
                                               training_time_limit=3600000, 
                                               loss = MeanPredLoss(),
                                               debug=False, sequence_splitter=lambda: ItemsMasking(masking_prob=0.2), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=relative_position_encoding),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               )
        return recommender


def bert4rec(relative_position_encoding, sequence_len=50):
        model = BERT4Rec(embedding_size=64, intermediate_size=128, num_hidden_layers=2, max_history_len=sequence_len)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=256,
                                               training_time_limit=3600000, 
                                               loss = MeanPredLoss(),
                                               debug=False, sequence_splitter=lambda: ItemsMasking(masking_prob=0.2), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=relative_position_encoding),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               )
        return recommender
recommenders = {
    "deberta4rec_relative-50": lambda: deberta4rec(True, 50), 
    "deberta4rec_static-50": lambda: deberta4rec(False, 50), 
    "bert4rec_relative-50": lambda: bert4rec(True, 50), 
    "bert4rec_static-50": lambda: bert4rec(False, 50), 

    "deberta4rec_relative-100": lambda: deberta4rec(True, 100), 
    "deberta4rec_static-100": lambda: deberta4rec(False, 100), 
    "bert4rec_relative-100": lambda: bert4rec(True, 100), 
    "bert4rec_static-100": lambda: bert4rec(False, 100), 

    "deberta4rec_relative-200": lambda: deberta4rec(True, 200), 
    "deberta4rec_static-200": lambda: deberta4rec(False, 200), 
    "bert4rec_relative-200": lambda: bert4rec(True, 200), 
    "bert4rec_static-200": lambda: bert4rec(False, 200), 

    "deberta4rec_relative-400": lambda: deberta4rec(True, 400), 
    "deberta4rec_static-400": lambda: deberta4rec(False, 400), 
    "bert4rec_relative-400": lambda: bert4rec(True, 400), 
    "bert4rec_static-400": lambda: bert4rec(False, 400), 
}

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), HIT(4), NDCG(40), MAP(10)]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)

def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

