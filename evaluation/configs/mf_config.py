from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.matrix_factorization import MatrixFactorizationRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
from aprec.evaluation.metrics.sps import SPS


DATASET = get_movielens_actions(min_rating=1.0)

USERS_FRACTIONS = [1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def mf_recommender(embedding_size, num_epochs, loss, batch_size):
    return FilterSeenRecommender(MatrixFactorizationRecommender(embedding_size, num_epochs, loss, batch_size))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

RECOMMENDERS = {
    "top_recommender": top_recommender, 
    "mf_32_30_lambdarank": lambda: mf_recommender(32, 30, 'lambdarank', 64),
    "mf_32_30_bpr": lambda: mf_recommender(32, 30, 'bpr', 64),
    "mf_32_30_bce": lambda: mf_recommender(32, 30, 'binary_crossentropy', 64),
    "mf_32_30_xendcg": lambda: mf_recommender(32, 30, 'xendcg', 64),
    "svd_recommender_32": lambda: svd_recommender(32),
    "lightfm_recommender_30_bpr": lambda: lightfm_recommender(30, 'bpr'),
#    "lightfm_recommender_30_warp": lambda: lightfm_recommender(30, 'warp'),
#    "lightfm_recommender_30_warp_kos": lambda: lightfm_recommender(30, 'warp-kos'),
#    "lightfm_recommender_100_bpr": lambda: lightfm_recommender(100, 'bpr'),
#    "lightfm_recommender_100_warp": lambda: lightfm_recommender(100, 'warp'),
#    "lightfm_recommender_100_warp_kos": lambda: lightfm_recommender(100, 'warp-kos'),
}

FRACTION_TO_SPLIT = 0.85

METRICS = [Precision(5), NDCG(40), Recall(5), SPS(10), MRR(), MAP(10)]


RECOMMENDATIONS_LIMIT = 100
SPLIT_STRATEGY = "LEAVE_ONE_OUT"

