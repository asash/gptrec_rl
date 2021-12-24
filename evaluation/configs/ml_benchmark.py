from aprec.datasets.movielens100k import get_movielens100k_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.salrec.salrec_recommender import SalrecRecommender
from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from tensorflow.keras.optimizers import Adam
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = "ml-100k"

USERS_FRACTIONS = [1.]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def vanilla_bert4rec(num_steps):
    recommender = VanillaBERT4Rec(num_train_steps=num_steps)
    return FilterSeenRecommender(recommender)

def salrec(loss, activation_override=None):
    activation = 'linear' if loss == 'lambdarank' else 'sigmoid'
    if activation_override is not None:
        activation = activation_override
    return FilterSeenRecommender(SalrecRecommender(train_epochs=10000, loss=loss,
                                                   optimizer=Adam(), early_stop_epochs=100,
                                                   batch_size=64, sigma=1.0, ndcg_at=10,
                                                   max_history_len=150,
                                                   output_layer_activation=activation,
                                                   num_blocks=2,
                                                   num_target_predictions=1
                                                   ))

def constant_recommender():
    return ConstantRecommender([('457', 0.45),
                                ('380', 0.414),
                                ('110', 0.413),
                                ('292', 0.365),
                                ('296', 0.323),
                                ('595', 0.313),
                                ('588', 0.312),
                                ('592', 0.293),
                                ('440', 0.286),
                                ('357', 0.286),
                                ('434', 0.280),
                                ('593', 0.280),
                                ('733', 0.276),
                                ('553', 0.257),
                                ('253', 0.257)])

RECOMMENDERS = {
    "Salrec-Lambdarank": lambda: salrec('lambdarank'),
    "top_recommender": top_recommender,
    "svd_recommender_30": lambda: svd_recommender(30),
    "Salrec-BCE": lambda: salrec('binary_crossentropy'),
    "Salrec-BPR": lambda: salrec('bpr', 'linear'),
    "constant_recommender": constant_recommender,
}

N_VAL_USERS=64
MAX_TEST_USERS=943

METRICS = [NDCG(10), Precision(5), NDCG(40), Recall(5), HIT(10), MRR(), MAP(10)]

SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
