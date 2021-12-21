from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.hit import HIT
from aprec.datasets.bert4rec_datasets import get_bert4rec_dataset
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.top_recommender import TopRecommender
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = get_bert4rec_dataset("ml-1m")

USERS_FRACTIONS = [1]


def vanilla_bert4rec(num_steps):
    recommender = VanillaBERT4Rec()
    return FilterSeenRecommender(recommender)

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def top_recommender_no_filters():
    return TopRecommender()

RECOMMENDERS = {
    "top_recommender": top_recommender,
    "top_recommender_no_filters": top_recommender_no_filters,
    "vanilla_bert4rec-400000": lambda: vanilla_bert4rec(400000),
    "vanilla_bert4rec-800000": lambda: vanilla_bert4rec(800000),
    "vanilla_bert4rec-1600000": lambda: vanilla_bert4rec(1600000),
    "vanilla_bert4rec-3200000": lambda: vanilla_bert4rec(1600000),
}


MAX_TEST_USERS=943

METRICS = [HIT(1), HIT(5), HIT(10), MRR()]


SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
