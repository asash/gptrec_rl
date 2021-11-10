from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.average_popularity_rank import AveragePopularityRank
from aprec.evaluation.metrics.pairwise_cos_sim import PairwiseCosSim
from aprec.evaluation.metrics.sps import SPS


DATASET = get_movielens_actions(min_rating=1.0)

USERS_FRACTIONS = [.1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

RECOMMENDERS = {
    "top_recommender": top_recommender, 
#    "svd_recommender_10": lambda: svd_recommender(10), 
#    "svd_recommender_20": lambda: svd_recommender(20), 
    "svd_recommender_30": lambda: svd_recommender(2),
    "svd_recommender_30": lambda: svd_recommender(4),
    "svd_recommender_30": lambda: svd_recommender(8),
    "svd_recommender_30": lambda: svd_recommender(16),
    "svd_recommender_30": lambda: svd_recommender(32),
    "svd_recommender_30": lambda: svd_recommender(64),
    "svd_recommender_30": lambda: svd_recommender(128),
    "svd_recommender_30": lambda: svd_recommender(256),
#    "lightfm_recommender_30_bpr": lambda: lightfm_recommender(30, 'bpr'),
#    "lightfm_recommender_30_warp": lambda: lightfm_recommender(30, 'warp'),
#    "lightfm_recommender_30_warp_kos": lambda: lightfm_recommender(30, 'warp-kos'),
#    "lightfm_recommender_100_bpr": lambda: lightfm_recommender(100, 'bpr'),
#    "lightfm_recommender_100_warp": lambda: lightfm_recommender(100, 'warp'),
#    "lightfm_recommender_100_warp_kos": lambda: lightfm_recommender(100, 'warp-kos'),
}

FRACTION_TO_SPLIT = 0.85

dataset_for_metric = [action for action in get_movielens_actions(min_rating=1.0)]
METRICS = [Recall(10), Recall(20), Recall(30), Recall(100), Recall(200), Recall(300), Recall(500), Recall(1000), Recall(2000), Recall(3000)]
del(dataset_for_metric)


RECOMMENDATIONS_LIMIT = 3000
SPLIT_STRATEGY = "LEAVE_ONE_OUT"

