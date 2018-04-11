from aprec.datasets.movielens import get_movielens_actions
from aprec.recommenders.top_recommender import TopRecommender
from aprec.utils import generator_limit
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.sps import SPS


DATASET = generator_limit(get_movielens_actions(min_rating=4.0), 1000_000)
RECOMMENDERS = {
    "top_recommender": TopRecommender    
}

FRACTIONS_TO_SPLIT = (0.7, )
METRICS = [Precision(10), NDCG(50), Recall(10), SPS(10)]

