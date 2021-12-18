from aprec.datasets.movielens20m import get_movielens20m_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.recall import Recall


DATASET = get_movielens20m_actions(min_rating=1.0)

USERS_FRACTIONS = [.1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

RECOMMENDERS = {
    "top_recommender": top_recommender, 
    "svd_recommender_2": lambda: svd_recommender(2),
    "svd_recommender_4": lambda: svd_recommender(4),
    "svd_recommender_8": lambda: svd_recommender(8),
    "svd_recommender_16": lambda: svd_recommender(16),
    "svd_recommender_32": lambda: svd_recommender(32),
    "svd_recommender_64": lambda: svd_recommender(64),
    "svd_recommender_128": lambda: svd_recommender(128),
    "svd_recommender_256": lambda: svd_recommender(256),
}

FRACTION_TO_SPLIT = 0.85

dataset_for_metric = [action for action in get_movielens20m_actions(min_rating=1.0)]
METRICS = [Recall(10), Recall(20), Recall(30), Recall(100), Recall(200), Recall(300), Recall(500), Recall(1000), Recall(2000), Recall(3000)]
del(dataset_for_metric)


RECOMMENDATIONS_LIMIT = 3000
SPLIT_STRATEGY = "LEAVE_ONE_OUT"

