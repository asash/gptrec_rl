from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = get_bert4rec_dataset("ml-1m")

USERS_FRACTIONS = [1]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

RECOMMENDERS = {
    "top_recommender": top_recommender,

}


METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR()]

SAMPLED_METRICS_ON=101
RECOMMENDATIONS_LIMIT = 900
SPLIT_STRATEGY = LeaveOneOut(max_test_users=6040, random_seed=2)
