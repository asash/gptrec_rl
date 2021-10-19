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
    "top_recommender": top_recommender, 
#    "svd_recommender_10": lambda: svd_recommender(10), 
#    "svd_recommender_20": lambda: svd_recommender(20), 
    "svd_recommender_30": lambda: svd_recommender(30),
    "lightfm_recommender_30_bpr": lambda: lightfm_recommender(30, 'bpr'),
    "lightfm_recommender_30_warp": lambda: lightfm_recommender(30, 'warp'),
    "lightfm_recommender_30_warp_kos": lambda: lightfm_recommender(30, 'warp-kos'),
    "lightfm_recommender_100_bpr": lambda: lightfm_recommender(100, 'bpr'),
    "lightfm_recommender_100_warp": lambda: lightfm_recommender(100, 'warp'),
    "lightfm_recommender_100_warp_kos": lambda: lightfm_recommender(100, 'warp-kos'),
    "constant_recommender": constant_recommender,
}

FRACTION_TO_SPLIT = 0.85

dataset_for_metric = [action for action in get_movielens_actions(min_rating=1.0)]
METRICS = [Precision(5), NDCG(40), Recall(5), SPS(10), MRR(), MAP(10), AveragePopularityRank(5, dataset_for_metric),
           PairwiseCosSim(dataset_for_metric, 10)]
del(dataset_for_metric)


SPLIT_STRATEGY = "LEAVE_ONE_OUT"

