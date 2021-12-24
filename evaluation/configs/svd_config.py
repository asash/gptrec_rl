from aprec.evaluation.configs.top_recommender_config import TARGET_ITEMS_SAMPLER
from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.lightfm import LightFMRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.split_actions import LeaveOneOut

from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR


DATASET = "BERT4rec.steam"

USERS_FRACTIONS = [1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss, num_threads=32))

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "svd_recommender_128": lambda: svd_recommender(128),
    "lightfm_recommeder_30":lambda: lightfm_recommender(128, 'bpr')
}

TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)
METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR()]
SPLIT_STRATEGY = LeaveOneOut(max_test_users=6040, random_seed=2)

