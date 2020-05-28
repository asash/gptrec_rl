from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.utils.generator_limit import generator_limit
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.sps import SPS


DATASET = get_movielens_actions(min_rating=1.0)
USERS_FRACTION = 0.05 

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

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
    "svd_recommender_10": lambda: svd_recommender(10), 
    "svd_recommender_20": lambda: svd_recommender(20), 
    "svd_recommender_30": lambda: svd_recommender(30), 
    "constant_recommender": constant_recommender,
}

FRACTIONS_TO_SPLIT = (0.85, )
METRICS = [Precision(5), NDCG(40), Recall(5), SPS(10)]

