from aprec.recommenders.top_recommender import TopRecommender
from aprec.recommenders.svd import SvdRecommender
from aprec.recommenders.constant_recommender import ConstantRecommender
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.recommenders.item_item import ItemItemRecommender
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.split_actions import RandomSplit




DATASET = "ml-20m"
USERS_FRACTIONS = [0.01]

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def svd_recommender(k):
    return FilterSeenRecommender(SvdRecommender(k))

def item_item_recommender():
    return FilterSeenRecommender(ItemItemRecommender())


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
    "item_item_recommender": item_item_recommender,
    "top_recommender": top_recommender, 
    "svd_recommender_40": lambda: svd_recommender(40), 
    "constant_recommender": constant_recommender,
}

METRICS = [Precision(5), NDCG(40), Recall(5), HIT(10)]

SPLIT_STRATEGY = RandomSplit()


